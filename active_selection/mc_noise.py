import math
from dataloaders.dataset import paths_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dataloaders.utils import map_segmentation_to_colors
import numpy as np
from active_selection.base import ActiveSelectionBase
from active_selection.mc_dropout import ActiveSelectionMCDropout
from tqdm import tqdm
import constants
from scipy import stats


class ActiveSelectionMCNoise(ActiveSelectionBase):

    def __init__(self, num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionMCNoise, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.dataset_num_classes = num_classes

    def _get_vote_entropy_for_batch_with_input_noise(self, model, image_batch, label_batch):

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                noise = np.random.normal(loc=0.0, scale=0.125, size=image_batch.shape).astype(np.float32)
                outputs[:, step, :, :] = torch.argmax(model(image_batch + torch.from_numpy(noise).cuda()), dim=1)

        entropy_maps = []

        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)
            mask = (label_batch[i, :, :] < 0) | (label_batch[i, :, :] >= self.dataset_num_classes)
            for c in range(self.dataset_num_classes):
                p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
                entropy_map = entropy_map - (p * torch.log2(p + 1e-12))
            entropy_map[mask] = 0
            # visualize for debugging

            # prediction = stats.mode(outputs[i, :, :, :].cpu().numpy(), axis=0)[0].squeeze()
            # self._visualize_entropy(image_batch[i, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
            entropy_maps.append(entropy_map)

        return entropy_maps

    def get_vote_entropy_for_images_with_input_noise(self, model, images, selection_count):

        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()

        entropies = []
        for sample in tqdm(loader):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            entropies.extend([torch.sum(x).cpu().item() / (image_batch.shape[2] * image_batch.shape[3])
                              for x in self._get_vote_entropy_for_batch_with_input_noise(model, image_batch, label_batch)])

        selected_samples = list(zip(*sorted(zip(entropies, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def _get_vote_entropy_for_batch_with_feature_noise(self, model, image_batch, label_batch):
        model.module.set_noisy_features(True)
        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                outputs[:, step, :, :] = torch.argmax(model(image_batch), dim=1)

        entropy_maps = []

        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)
            mask = (label_batch[i, :, :] < 0) | (label_batch[i, :, :] >= self.dataset_num_classes)
            for c in range(self.dataset_num_classes):
                p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
                entropy_map = entropy_map - (p * torch.log2(p + 1e-12))
            entropy_map[mask] = 0
            # visualize for debugging

            #prediction = stats.mode(outputs[i, :, :, :].cpu().numpy(), axis=0)[0].squeeze()
            #self._visualize_entropy(image_batch[i, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
            entropy_maps.append(entropy_map)
        model.module.set_noisy_features(False)
        return entropy_maps

    def _get_vote_entropy_for_batch_with_mc_dropout(self, model, image_batch, label_batch):

        def turn_on_dropout(m):
            if type(m) == torch.nn.Dropout2d:
                m.train()
        model.apply(turn_on_dropout)

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                outputs[:, step, :, :] = torch.argmax(model(image_batch), dim=1)

        entropy_maps = []

        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)
            mask = (label_batch[i, :, :] < 0) | (label_batch[i, :, :] >= self.dataset_num_classes)
            for c in range(self.dataset_num_classes):
                p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
                entropy_map = entropy_map - (p * torch.log2(p + 1e-12))
            entropy_map[mask] = 0
            # visualize for debugging
            # prediction = stats.mode(outputs[i, :, :, :].cpu().numpy(), axis=0)[0].squeeze()
            # self._visualize_entropy(image_batch[i, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
            entropy_maps.append(entropy_map)

        model.eval()

        return entropy_maps

    def get_vote_entropy_for_images_with_feature_noise(self, model, images, selection_count):

        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()
        entropies = []
        for sample in tqdm(loader):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            entropies.extend([torch.sum(x).cpu().item() / (image_batch.shape[2] * image_batch.shape[3])
                              for x in self._get_vote_entropy_for_batch_with_feature_noise(model, image_batch, label_batch)])

        selected_samples = list(zip(*sorted(zip(entropies, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def get_vote_entropy_for_batch_with_noise_and_vote_entropy(self, model, images, selection_count):

        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()

        entropies = []
        for sample in tqdm(loader):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            noise_entropies = self._get_vote_entropy_for_batch_with_feature_noise(model, image_batch, label_batch)
            mc_entropies = self._get_vote_entropy_for_batch_with_mc_dropout(model, image_batch, label_batch)
            combined_entropies = [x + y for x, y in zip(noise_entropies, mc_entropies)]
            entropies.extend([torch.sum(x).cpu().item() / (image_batch.shape[2] * image_batch.shape[3])
                              for x in combined_entropies])

        selected_samples = list(zip(*sorted(zip(entropies, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]

        return selected_samples

    def create_region_maps(self, model, images, existing_regions, region_size, selection_size):
        base_size = 512 if self.crop_size == -1 else self.crop_size
        score_maps = torch.cuda.FloatTensor(len(images), base_size - region_size + 1, base_size - region_size + 1)
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        weights = torch.cuda.FloatTensor(region_size, region_size).fill_(1.)

        map_ctr = 0
        # commented lines are for visualization and verification
        # entropy_maps = []
        # base_images = []
        for sample in tqdm(loader):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            noise_entropies = self._get_vote_entropy_for_batch_with_feature_noise(model, image_batch, label_batch)
            mc_entropies = self._get_vote_entropy_for_batch_with_mc_dropout(model, image_batch, label_batch)
            combined_entropies = [x + y for x, y in zip(noise_entropies, mc_entropies)]
            for img_idx, entropy_map in enumerate(combined_entropies):
                ActiveSelectionMCDropout.suppress_labeled_entropy(entropy_map, existing_regions[map_ctr])
                # base_images.append(image_batch[img_idx, :, :, :].cpu().numpy())
                # entropy_maps.append(entropy_map.cpu().numpy())
                score_maps[map_ctr, :, :] = torch.nn.functional.conv2d(entropy_map.unsqueeze(
                    0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
                map_ctr += 1

        min_val = score_maps.min()
        max_val = score_maps.max()
        minmax_norm = lambda x: x.add_(-min_val).mul_(1.0 / (max_val - min_val))
        minmax_norm(score_maps)

        num_requested_indices = (selection_size * base_size * base_size) / (region_size * region_size)
        regions, num_selected_indices = ActiveSelectionMCDropout.square_nms(score_maps.cpu(), region_size, num_requested_indices)
        # print(f'Requested/Selected indices {num_requested_indices}/{num_selected_indices}')

        # for i in range(len(regions)):
        #    ActiveSelectionMCDropout._visualize_regions(base_images[i], entropy_maps[i], regions[i], region_size)

        new_regions = {}
        for i in range(len(regions)):
            if regions[i] != []:
                new_regions[images[i]] = regions[i]

        model.eval()

        return new_regions, num_selected_indices

    @staticmethod
    def _visualize_entropy(image_normalized, entropy_map, prediction):
        import matplotlib
        import matplotlib.pyplot as plt
        image_unnormalized = ((np.transpose(image_normalized, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
        prediction_mapped = map_segmentation_to_colors(prediction.astype(np.uint8), 'cityscapes')
        norm = matplotlib.colors.Normalize(vmin=np.min(entropy_map), vmax=np.max(entropy_map), clip=False)
        plt.figure()
        plt.title('display')
        plt.subplot(1, 3, 1)
        plt.imshow(image_unnormalized)
        plt.subplot(1, 3, 2)
        plt.imshow(prediction_mapped)
        plt.subplot(1, 3, 3)
        plt.imshow(entropy_map, norm=norm, cmap='gray')
        plt.show(block=True)
