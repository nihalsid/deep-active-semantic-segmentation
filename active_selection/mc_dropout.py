import math
from dataloaders.dataset import paths_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dataloaders.utils import map_segmentation_to_colors
import numpy as np
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm
import constants


class ActiveSelectionMCDropout(ActiveSelectionBase):

    def __init__(self, dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionMCDropout, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.dataset_num_classes = dataset_num_classes

    def get_random_uncertainity(self, images, selection_count):
        scores = []
        for i in range(len(images)):
            scores.append(random.random())
        selected_samples = list(zip(*sorted(zip(scores, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def _get_vote_entropy_for_batch(self, model, image_batch):

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                outputs[:, step, :, :] = torch.argmax(model(image_batch), dim=1)

        entropy_maps = []

        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)

            for c in range(self.dataset_num_classes):
                p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
                entropy_map = entropy_map - (p * torch.log2(p + 1e-12))

            # visualize for debugging
            # prediction = stats.mode(outputs[i, :, :, :].cpu().numpy(), axis=0)[0].squeeze()
            # self._visualize_entropy(image_batch[i, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
            entropy_maps.append(entropy_map)

        return entropy_maps

    @staticmethod
    def square_nms(score_maps, region_size, max_selection_count):
        ones_tensor = torch.FloatTensor(score_maps.shape[1], score_maps.shape[2]).fill_(0)
        selected_regions = [[] for x in range(score_maps.shape[0])]

        tbar = tqdm(list(range(math.ceil(max_selection_count))), desc='NMS')

        selection_count = 0
        for iter_idx in tbar:
            argmax = score_maps.view(-1).argmax()
            i, r, c = argmax // (score_maps.shape[1] * score_maps.shape[2]), (argmax //
                                                                              score_maps.shape[2]) % score_maps.shape[1], argmax % score_maps.shape[2]

            selected_regions[i.item()].append((r.item(), c.item(), region_size, region_size))
            selection_count += 1
            zero_out_mask = ones_tensor != 0
            r0 = max(0, r - region_size)
            c0 = max(0, c - region_size)
            r1 = min(score_maps.shape[1], r + region_size)
            c1 = min(score_maps.shape[2], c + region_size)
            zero_out_mask[r0:r1, c0:c1] = 1
            score_maps[i, zero_out_mask] = 0

            if score_maps.max() < 0.1:
                break

        return selected_regions, selection_count

    @staticmethod
    def suppress_labeled_entropy(entropy_map, labeled_region):
        ones_tensor = torch.cuda.FloatTensor(entropy_map.shape[0], entropy_map.shape[1]).fill_(0)
        if labeled_region:
            for lr in labeled_region:
                zero_out_mask = ones_tensor != 0
                r0 = lr[0]
                c0 = lr[1]
                r1 = lr[0] + lr[2]
                c1 = lr[1] + lr[3]
                zero_out_mask[r0:r1, c0:c1] = 1
                entropy_map[zero_out_mask] = 0

    def create_region_maps(self, model, images, existing_regions, region_size, selection_size):

        def turn_on_dropout(m):
            if type(m) == torch.nn.Dropout2d:
                m.train()
        model.apply(turn_on_dropout)

        score_maps = torch.cuda.FloatTensor(len(images), self.crop_size - region_size + 1, self.crop_size - region_size + 1)
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        weights = torch.cuda.FloatTensor(region_size, region_size).fill_(1.)

        map_ctr = 0
        # commented lines are for visualization and verification
        # entropy_maps = []
        # base_images = []
        for image_batch in tqdm(loader):
            image_batch = image_batch.cuda()
            for img_idx, entropy_map in enumerate(self._get_vote_entropy_for_batch(model, image_batch)):
                ActiveSelectionMCDropout.suppress_labeled_entropy(entropy_map, existing_regions[map_ctr])
                # base_images.append(image_batch[img_idx, :, :, :].cpu().numpy())
                # entropy_maps.append(entropy_map.cpu().numpy())
                score_maps[map_ctr, :, :] = torch.nn.functional.conv2d(entropy_map.unsqueeze(
                    0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
                map_ctr += 1

        min_val = score_maps.min()
        max_val = score_maps.max()
        minmax_norm = lambda x: x.add_(min_val).mul_(1.0 / (max_val - min_val))
        minmax_norm(score_maps)

        num_requested_indices = (selection_size * self.crop_size * self.crop_size) / (region_size * region_size)
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

    def get_vote_entropy_for_images(self, model, images, selection_count):

        def turn_on_dropout(m):
            if type(m) == torch.nn.Dropout2d:
                m.train()
        model.apply(turn_on_dropout)

        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size), batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)

        entropies = []
        for image_batch in tqdm(loader):
            image_batch = image_batch.cuda()
            entropies.extend([torch.sum(x).cpu().item() / (image_batch.shape[2] * image_batch.shape[3])
                              for x in self._get_vote_entropy_for_batch(model, image_batch)])

        model.eval()
        selected_samples = list(zip(*sorted(zip(entropies, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

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

    @staticmethod
    def _visualize_regions(base_image, image, regions, region_size):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        _, ax = plt.subplots(2)
        if len(base_image.shape) == 3:
            ax[0].imshow(((np.transpose(base_image, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8))
        else:
            ax[0].imshow(base_image)
        ax[1].imshow(image)
        for r in regions:
            rect = patches.Rectangle((r[1], r[0]), region_size, region_size, linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)