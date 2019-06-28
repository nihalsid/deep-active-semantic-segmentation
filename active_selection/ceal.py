from dataloaders.dataset import paths_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm
import matplotlib
from dataloaders.utils import map_segmentation_to_colors


class ActiveSelectionCEAL(ActiveSelectionBase):

    def __init__(self, dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionCEAL, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.dataset_num_classes = dataset_num_classes

    def get_least_confident_samples(self, model, images, selection_count):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        max_confidence = []

        #rgb_images = []
        #sem_gt_images = []
        #sem_pred_images = []
        #lc_images = []

        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                softmax = torch.nn.Softmax2d()
                output = model(image_batch)
                max_conf_batch = torch.max(softmax(output), dim=1)[0]
                for batch_idx in range(max_conf_batch.shape[0]):
                    mask = (label_batch[batch_idx, :, :] < 0) | (label_batch[batch_idx, :, :] >= self.dataset_num_classes)
                    max_conf_batch[batch_idx, mask] = 1

                    # from active_selection import ActiveSelectionMCDropout
                    # ActiveSelectionMCDropout._visualize_entropy(image_batch[batch_idx, :, :, :].cpu().numpy(), max_conf_batch[
                    #                                            batch_idx, :, :].cpu().numpy(), prediction)
                    '''
                    image_unnormalized = ((np.transpose(image_batch[batch_idx].cpu().numpy(), axes=[1, 2, 0])
                                           * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)

                    rgb_images.append(image_unnormalized)
                    gt_colored = map_segmentation_to_colors(np.array(label_batch[batch_idx].cpu().numpy()).astype(np.uint8), 'cityscapes')
                    sem_gt_images.append(gt_colored)
                    prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0)
                    sem_pred_images.append(map_segmentation_to_colors(np.array(prediction).astype(np.uint8), 'cityscapes'))
                    masked_target_array = np.ma.array(max_conf_batch[batch_idx, :, :].cpu().numpy(), mask=label_batch[batch_idx].cpu().numpy() == 255)
                    masked_target_array = 1 - masked_target_array
                    cmap = matplotlib.cm.jet
                    cmap.set_bad('white', 1.)
                    lc_images.append(cmap(masked_target_array))
                    '''
                    max_confidence.append(torch.mean(max_conf_batch[batch_idx, :, :]).cpu().item())
        '''
        import matplotlib.pyplot as plt
        for prefix, arr in zip(['rgb', 'sem_gt', 'sem_pred', 'lc'], [rgb_images, sem_gt_images, sem_pred_images, lc_images]):
            stacked_image = np.ones(((arr[0].shape[0] + 20) * len(arr), arr[0].shape[1], arr[0].shape[2]),
                                    dtype=arr[0].dtype) * (255 if arr[0].dtype == np.uint8 else 1)
            for i, im in enumerate(arr):
                stacked_image[i * (arr[0].shape[0] + 20): i * (arr[0].shape[0] + 20) + arr[0].shape[0], :, :] = im
            plt.imsave('%s.png' % (prefix), stacked_image)
        '''
        selected_samples = list(zip(*sorted(zip(max_confidence, images), key=lambda x: x[0], reverse=False)))[1][:selection_count]
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
                    mask = (label_batch[batch_idx, :, :] < 0) | (label_batch[batch_idx, :, :] >= self.dataset_num_classes)
                    mask = mask.cpu().numpy().astype(np.bool)
                    most_confident_scores = torch.max(output[batch_idx, :, :].squeeze(), dim=0)[0].cpu().numpy()
                    output_numpy = output[batch_idx, :, :, :].cpu().numpy()
                    ndx = np.indices(output_numpy.shape)
                    second_most_confident_scores = output_numpy[output_numpy.argsort(0), ndx[1], ndx[2]][-2]
                    margin = most_confident_scores - second_most_confident_scores
                    margin[mask] = 1
                    # from active_selection import ActiveSelectionMCDropout
                    # prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0)
                    # ActiveSelectionMCDropout._visualize_entropy(image_batch[batch_idx, :, :, :].cpu().numpy(), margin, prediction)
                    margins.append(np.mean(margin))

        selected_samples = list(zip(*sorted(zip(margins, images), key=lambda x: x[0], reverse=False)))[1][:selection_count]
        return selected_samples

    def _get_entropies(self, model, images):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        entropies = []

        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                softmax = torch.nn.Softmax2d()
                output = softmax(model(image_batch))
                num_classes = output.shape[1]

                for batch_idx in range(output.shape[0]):
                    mask = (label_batch[batch_idx, :, :] < 0) | (label_batch[batch_idx, :, :] >= self.dataset_num_classes)
                    entropy_map = torch.cuda.FloatTensor(output.shape[2], output.shape[3]).fill_(0)
                    for c in range(num_classes):
                        entropy_map = entropy_map - (output[batch_idx, c, :, :] * torch.log2(output[batch_idx, c, :, :] + 1e-12))
                    entropy_map[mask] = 0
                    # from active_selection import ActiveSelectionMCDropout
                    # prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0)
                    # ActiveSelectionMCDropout._visualize_entropy(image_batch[batch_idx, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
                    entropies.append(np.mean(entropy_map.cpu().numpy()))

        return entropies

    def get_maximum_entropy_samples(self, model, images, selection_count):
        entropies = self._get_entropies(model, images)
        selected_samples = list(zip(*sorted(zip(entropies, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples, entropies

    def get_fusion_of_confidence_margin_entropy_samples(self, model, images, selection_count):
        import random
        samples1 = self.get_least_confident_samples(model, images, selection_count)
        samples2 = self.get_least_margin_samples(model, images, selection_count)
        samples3 = self.get_maximum_entropy_samples(model, images, selection_count)[0]
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
                    mask = (label_batch[batch_idx, :, :] < 0) | (label_batch[batch_idx, :, :] >= self.dataset_num_classes)
                    mask = mask.cpu().numpy().astype(np.bool)
                    prediction = np.argmax(output[batch_idx, :, :, :].cpu().numpy().squeeze(), axis=0).astype(np.uint8)
                    prediction[mask] = 255
                    weak_labels.append(prediction)

        return dict(zip(selected_images, weak_labels))
