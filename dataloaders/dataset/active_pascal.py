import os
import numpy as np
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr
import random
import math
import torch
import pickle
from active_selection.mc_dropout import ActiveSelectionMCDropout
from dataloaders.dataset import pascal_base
from torch.utils import data
import constants
from tqdm import tqdm


class ActivePascalImage(pascal_base.ActivePascalBase):

    def __init__(self, path, base_size, crop_size, split, init_set, overfit=False, memory_hog_mode=True):

        super(ActivePascalImage, self).__init__(path, base_size, crop_size, split, overfit)
        self.current_image_paths = self.image_paths
        self.remaining_image_paths = []
        if self.split == 'train':
            with open(os.path.join(self.path, 'seed_sets', init_set), "r") as fptr:
                self.current_image_paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
                self.remaining_image_paths = [x for x in self.image_paths if x not in self.current_image_paths]
                print(f'# of current_image_paths = {len(self.current_image_paths)}, # of remaining_image_paths = {len(self.remaining_image_paths)}')

        self.labeled_pixel_count = len(self.current_image_paths) * self.base_size * self.base_size
        self.memory_hog_mode = memory_hog_mode
        if self.memory_hog_mode:
            self.path_to_npy = {}
            self.load_files_into_memory()

    def load_files_into_memory(self):
        print('Acquiring dataset in memory')
        for n in tqdm(self.current_image_paths):
            if n not in self.path_to_npy:
                with self.env.begin(write=False) as txn:
                    loaded_npy = pickle.loads(txn.get(n))
                    self.path_to_npy[n] = loaded_npy

    def __getitem__(self, index):

        img_path = None

        is_weakly_labeled = False

        if index >= len(self.current_image_paths):
            is_weakly_labeled = True

        img_path = self.current_image_paths[index] if not is_weakly_labeled else self.weakly_labeled_image_paths[index - len(self.current_image_paths)]

        assert not (img_path in self.weakly_labeled_image_paths and img_path in self.current_image_paths), "weakly labeled image exists in already labeled samples"

        loaded_npy = None

        if self.memory_hog_mode and img_path in self.path_to_npy:
            loaded_npy = self.path_to_npy[img_path]
        else:
            with self.env.begin(write=False) as txn:
                loaded_npy = pickle.loads(txn.get(img_path))

        image = loaded_npy[:, :, 0:3]
        retval = None

        if is_weakly_labeled:
            target = self.weakly_labeled_targets[img_path]
            retval = self.transform_val({'image': image, 'label': loaded_npy[:, :, 3]})
            retval['label'] = torch.from_numpy(target.astype(np.float32)).float()
        else:
            target = loaded_npy[:, :, 3]
            sample = {'image': image, 'label': target}
            retval = self.get_transformed_sample(sample)

        return retval

    def expand_training_set(self, paths):
        self.current_image_paths.extend(paths)
        for x in paths:
            self.remaining_image_paths.remove(x)
        self.labeled_pixel_count = len(self.current_image_paths) * self.base_size * self.base_size

    def add_weak_labels(self, predictions_dict):
        print(f'Adding {len(predictions_dict.keys())} weak labels')
        self.weakly_labeled_image_paths = list(predictions_dict.keys())
        self.weakly_labeled_targets = predictions_dict

    def clear_weak_labels(self):
        self.weakly_labeled_targets = {}
        self.weakly_labeled_image_paths = []


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentation_to_colors
    path = os.path.join(constants.DATASET_ROOT, 'pascal')
    crop_size = -1
    base_size = 512
    split = 'train'

    pascal_train = ActivePascalImage(path, base_size, crop_size, split, 'set_0.txt')
    dataloader = DataLoader(pascal_train, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader, 0):
        for j in range(sample['image'].size()[0]):
            print(pascal_train.current_image_paths[i])
            image = sample['image'].numpy()
            gt = sample['label'].numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'pascal')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(gt_colored)

        plt.show(block=True)
