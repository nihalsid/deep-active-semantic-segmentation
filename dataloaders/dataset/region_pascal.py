import pickle
import numpy as np
from PIL import Image
from dataloaders.dataset import pascal_base
from dataloaders.dataset import active_pascal
from collections import OrderedDict
from utils.cityscapes_to_lmdb import CITYSCAPES_IGNORE_INDEX
import constants
import os
from dataloaders import custom_transforms as tr
from tqdm import tqdm


class ActivePascalRegion(pascal_base.ActivePascalBase):

    # TODO: add support for weak labeling
    def __init__(self, path, base_size, crop_size, split, init_set, overfit=False, memory_hog_mode=True):

        super(ActivePascalRegion, self).__init__(path, base_size, crop_size, split, overfit)
        self.current_paths_to_regions_map = OrderedDict({})
        if self.split == 'train':
            with open(os.path.join(self.path, 'seed_sets', init_set), "r") as fptr:
                for path in fptr.readlines():
                    if path is not '':
                        path = u'{}'.format(path.strip()).encode('ascii')
                        self.current_paths_to_regions_map[path] = [(0, 0, crop_size, crop_size)]

        else:
            for path in self.image_paths:
                self.current_paths_to_regions_map[path] = [(0, 0, crop_size, crop_size)]

        self.memory_hog_mode = memory_hog_mode
        if self.memory_hog_mode:
            self.path_to_npy = {}
            self.load_files_into_memory()

        self._update_path_lists()
        self.labeled_pixel_count = crop_size * crop_size * len(self.current_image_paths)
        print(f'# of current_image_paths = {len(self.current_image_paths)}')

    def expand_training_set(self, new_regions, labeled_pixels):
        for path, regions in new_regions.items():
            if path in self.current_paths_to_regions_map:
                self.current_paths_to_regions_map[path].extend(regions)
            else:
                self.current_paths_to_regions_map[path] = regions
        self.labeled_pixel_count += labeled_pixels
        self._update_path_lists()

    def _update_path_lists(self):
        assert len(self.current_image_paths) == len(list(set(self.current_image_paths))), "updating expanded list"
        self.current_image_paths = list(self.current_paths_to_regions_map.keys())

    def get_existing_region_maps(self):
        regions = []
        for path in self.image_paths:
            if path in self.current_paths_to_regions_map:
                regions.append(self.current_paths_to_regions_map[path])
            else:
                regions.append([])
        return regions

    def load_files_into_memory(self):
        print('Acquiring dataset in memory')
        for n in tqdm(self.current_image_paths):
            if n not in self.path_to_npy:
                with self.env.begin(write=False) as txn:
                    loaded_npy = pickle.loads(txn.get(n))
                    self.path_to_npy[n] = loaded_npy

    def __getitem__(self, index):

        img_path = None
        regions = None

        img_path, regions = self.current_image_paths[index], self.current_paths_to_regions_map[self.current_image_paths[index]]

        loaded_npy = None
        if self.memory_hog_mode and img_path in self.path_to_npy:
            loaded_npy = self.path_to_npy[img_path]
        else:
            with self.env.begin(write=False) as txn:
                loaded_npy = pickle.loads(txn.get(img_path))

        image = loaded_npy[:, :, 0:3]
        target_full = loaded_npy[:, :, 3]

        target_masked = np.ones(target_full.shape, dtype=target_full.dtype) * CITYSCAPES_IGNORE_INDEX

        for r in regions:
            tr.invert_fix_scale_crop(target_full, target_masked, r, self.crop_size)

        sample = {'image': image, 'label': target_masked}
        return self.get_transformed_sample(sample)

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentation_to_colors

    path = os.path.join(constants.DATASET_ROOT, 'pascal')
    crop_size = 513
    base_size = 513
    split = 'train'

    pascal_train = ActivePascalRegion(path, base_size, crop_size, split, 'set_dummy.txt')
    pascal_train.expand_training_set({pascal_train.image_paths[51]: [(36, 100, 127, 127)]}, 0)
    dataloader = DataLoader(pascal_train, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        for j in range(sample['image'].size()[0]):
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
