from dataloaders.dataset import paths_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm


class ActiveSelectionCEAL(ActiveSelectionBase):

    def __init__(self, dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionCEAL, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.dataset_num_classes = dataset_num_classes

    def get_least_confident_samples(self, model, images, selection_count):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        max_confidence = []

        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                softmax = torch.nn.Softmax2d()
                output = model(image_batch)
                max_conf_batch = torch.max(softmax(output), dim=1)[0]
                for batch_idx in range(max_conf_batch.shape[0]):
                    mask = (label_batch[batch_idx, :, :] >= 0) & (label_batch[batch_idx, :, :] < self.dataset_num_classes)
                    # prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0)
                    # ActiveSelectionMCDropout._visualize_entropy(image_batch[batch_idx, :,
                    # :,:].cpu().numpy(), max_conf_batch[batch_idx, :, :].cpu().numpy(),
                    # prediction)
                    max_confidence.append(torch.sum(1 - max_conf_batch[batch_idx, mask]).cpu().item())

        selected_samples = list(zip(*sorted(zip(max_confidence, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def get_least_margin_samples(self, model, images, selection_count):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        margins = []
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                softmax = torch.nn.Softmax2d()
                output = softmax(model(image_batch))
                for batch_idx in range(output.shape[0]):
                    mask = (label_batch[batch_idx, :, :] >= 0) & (label_batch[batch_idx, :, :] < self.dataset_num_classes)
                    most_confident_scores = torch.max(output[batch_idx, :, mask].squeeze(), dim=0)[0].cpu().numpy()
                    output_numpy = output[batch_idx, :, :, :].cpu().numpy()
                    ndx = np.indices(output_numpy.shape)
                    second_most_confident_scores = output_numpy[output_numpy.argsort(0), ndx[1], ndx[2]][-2]
                    second_most_confident_scores = second_most_confident_scores[mask]
                    # prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0)
                    # ActiveSelectionMCDropout._visualize_entropy(image_batch[batch_idx, :, :,
                    # :].cpu().numpy(), most_confident_scores - second_most_confident_scores,
                    # prediction)
                    margins.append(np.sum(1 - (most_confident_scores - second_most_confident_scores)))

        selected_samples = list(zip(*sorted(zip(margins, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def _get_entropies(self, model, images):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size), batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        entropies = []

        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                softmax = torch.nn.Softmax2d()
                output = softmax(model(image_batch))
                num_classes = output.shape[1]

                for batch_idx in range(output.shape[0]):
                    mask = (label_batch[batch_idx, :, :] >= 0) & (label_batch[batch_idx, :, :] < self.dataset_num_classes)
                    entropy_map = torch.cuda.FloatTensor(output.shape[2], output.shape[3]).fill_(0)
                    for c in range(num_classes):
                        entropy_map = entropy_map - (output[batch_idx, c, :, :] * torch.log2(output[batch_idx, c, :, :] + 1e-12))
                    entropy_map[mask] = 0
                    # prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0)
                    # ActiveSelectionMCDropout._visualize_entropy(image_batch[batch_idx, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
                    entropies.append(np.sum(entropy_map.cpu().numpy()))

        return entropies

    def get_maximum_entropy_samples(self, model, images, selection_count):
        entropies = self._get_entropies(model, images)
        selected_samples = list(zip(*sorted(zip(entropies, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples, entropies

    def get_fusion_of_confidence_margin_entropy_samples(self, model, images, selection_count):
        import random
        samples1 = self.get_least_confident_samples(model, images, selection_count)
        samples2 = self.get_least_margin_samples(model, images, selection_count)
        samples3 = self.get_maximum_entropy_samples(model, images, selection_count)
        samples = list(set(samples1 + samples2 + samples3))
        random.shuffle(samples)
        return samples[:selection_count]

    def get_weakly_labeled_data(self, model, images, threshold, entropies=None):
        if not entropies:
            entropies = self._get_entropies(model, images)

        selected_images = []
        weak_labels = []
        for image, entropy in zip(images, entropies):
            if entropy < threshold:
                selected_images.append(image)

        loader = DataLoader(paths_dataset.PathsDataset(self.env, selected_images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                output = model(image_batch)
                for batch_idx in range(output.shape[0]):
                    mask = (label_batch[batch_idx, :, :] >= 0) & (label_batch[batch_idx, :, :] < self.dataset_num_classes)
                    prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0).astype(np.uint8)
                    prediction[mask] = 255
                    weak_labels.append(prediction)

        return dict(zip(selected_images, weak_labels))
