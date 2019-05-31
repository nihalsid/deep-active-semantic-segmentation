import numpy as np
from PIL import Image
from torch.utils import data
import glob
from pathlib import Path
import os
import constants
from dataloaders.dataset import cityscapes_base
import pickle


class Cityscapes(cityscapes_base.CityscapesBase):

    def __init__(self, path, base_size, crop_size, split, overfit=False):

        super(Cityscapes, self).__init__(path, base_size, crop_size, split, overfit)
        self.image_paths = self.image_paths[:60]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        img_path = self.image_paths[index]

        loaded_npy = None
        with self.env.begin(write=False) as txn:
            loaded_npy = pickle.loads(txn.get(img_path))

        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]

        sample = {'image': image, 'label': target}
        return self.get_transformed_sample(sample)

    def set_paths(self, pathlist):
        self.image_paths = pathlist

    def _fix_list_multiple_of_batch_size(self, paths, batch_size):
        remainder = len(paths) % batch_size
        if remainder != 0:
            num_new_entries = batch_size - remainder
            new_entries = paths[:num_new_entries]
            paths.extend(new_entries)
        return paths

    def make_dataset_multiple_of_batchsize(self, batch_size):
        self.original_size = len(self.image_paths)
        self.image_paths = self._fix_list_multiple_of_batch_size(self.image_paths, batch_size)

    def reset_dataset(self):
        self.image_paths = self.image_paths[:self.original_size_current]


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader

    torch.multiprocessing.set_start_method('spawn')
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentation_to_colors

    path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    crop_size = 513
    base_size = 513
    split = 'val'

    cityscapes_train = Cityscapes(path, base_size, crop_size, split)
    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=0)

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
