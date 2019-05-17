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
from scipy import stats


class ActiveSelectionMCNoise(ActiveSelectionBase):

    def __init__(self, num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionMCNoise, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.dataset_num_classes = num_classes

    def _get_vote_entropy_for_batch_with_input_noise(self, model, image_batch):

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                noise = np.random.normal(loc=0.0, scale=0.125, size=image_batch.shape).astype(np.float32)
                outputs[:, step, :, :] = torch.argmax(model(image_batch + torch.from_numpy(noise).cuda()), dim=1)

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

    def get_vote_entropy_for_images_with_input_noise(self, model, images, selection_count):

        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size), batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()

        entropies = []
        for image_batch in tqdm(loader):
            image_batch = image_batch.cuda()
            entropies.extend([torch.sum(x).cpu().item() / (image_batch.shape[2] * image_batch.shape[3])
                              for x in self._get_vote_entropy_for_batch_with_input_noise(model, image_batch)])

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
