import os
import numpy as np
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr
import glob
from pathlib import Path
import random
import math
import torch
import pickle
from utils.active_selection import ActiveSelectionMCDropout
from dataloaders.dataset import cityscapes_base
from dataloaders.dataset.cityscapes_base import Mode
from torch.utils import data
import constants


class ActiveCityscapesImage(cityscapes_base.ActiveCityscapesBase):

    def __init__(self, path, base_size, crop_size, split, init_set, overfit=False):

        super(ActiveCityscapesImage, self).__init__(path, base_size, crop_size, split, overfit)

        if self.split == 'train':
            self.current_image_paths = self.image_paths
            self.remaining_image_paths = []
            with open(os.path.join(self.path, 'seed_sets', init_set), "r") as fptr:
                self.current_image_paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
                self.remaining_image_paths = [x for x in self.image_paths if x not in self.current_image_paths]
                print(f'# of current_image_paths = {len(self.current_image_paths)}, # of remaining_image_paths = {len(self.remaining_image_paths)}')
        else:
            self.current_image_paths = self.image_paths
            self.remaining_image_paths = []
        self.labeled_pixel_count = len(self.current_image_paths) * self.crop_size * self.crop_size
        self.last_added_image_paths = self.current_image_paths.copy()

    def __getitem__(self, index):

        img_path = None

        if self.mode == Mode.ALL_BATCHES:
            img_path = self.current_image_paths[index]
        else:
            img_path = self.last_added_image_paths[index]

        loaded_npy = None
        with self.env.begin(write=False) as txn:
            loaded_npy = pickle.loads(txn.get(img_path))

        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]

        sample = {'image': Image.fromarray(image), 'label': Image.fromarray(target)}
        return self.get_transformed_sample(sample)

    def expand_training_set(self, scores, batch_size):
        num_new_samples = min(batch_size, len(scores))
        selected_samples = list(zip(*sorted(zip(scores, self.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:num_new_samples]
        self.current_image_paths.extend(selected_samples)
        self.last_added_image_paths = selected_samples
        for x in selected_samples:
            self.remaining_image_paths.remove(x)
        self.labeled_pixel_count = len(self.current_image_paths) * self.crop_size * self.crop_size

    def expand_training_set(self, paths):
        self.current_image_paths.extend(paths)
        self.last_added_image_paths = paths
        for x in paths:
            self.remaining_image_paths.remove(x)
        self.labeled_pixel_count = len(self.current_image_paths) * self.crop_size * self.crop_size


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentation_to_colors
    path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    crop_size = 513
    base_size = 513
    split = 'train'

    cityscapes_train = ActiveCityscapesImage(path, base_size, crop_size, split, 'set_0.txt')
    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=0)

    active_selector = ActiveSelectionMCDropout(19, cityscapes_train.env, crop_size, 2)
    print('Before Expansion', len(dataloader))
    cityscapes_train.expand_training_set(active_selector.get_random_uncertainity(cityscapes_train.current_image_paths), 10)
    print('After Expansion', len(dataloader))

    for i, sample in enumerate(dataloader, 0):
        for j in range(sample['image'].size()[0]):
            image = sample['image'].numpy()
            gt = sample['label'].numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(gt_colored)

        if i == 1:
            break

    plt.show(block=True)
