from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils import data
import lmdb
import os
from enum import Enum
import json
import random
import pickle


class CityscapesBase(data.Dataset):

    NUM_CLASSES = 19

    def __init__(self, path, base_size, crop_size, split, overfit):
        self.env = lmdb.open(os.path.join(path, split + ".db"), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.image_paths = pickle.loads(txn.get(b'__keys__'))

        self.path = path
        self.split = split
        self.crop_size = crop_size
        self.base_size = base_size
        self.overfit = overfit
        if crop_size == -1:
            self.scalecrop = tr.Scale(base_size=self.base_size)
        else:
            self.scalecrop = tr.FixScaleCrop(crop_size=self.crop_size)

        if overfit:
            self.image_paths = self.image_paths[:1]

        if len(self.image_paths) == 0:
            raise Exception("No images found in dataset directory")

    def transform_train(self, sample):

        composed_transforms = transforms.Compose([
            self.scalecrop,
            tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            self.scalecrop,
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            self.scalecrop,
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def get_transformed_sample(self, sample):
        retval = None

        if self.overfit:
            retval = self.transform_test(sample)

        if self.split == 'train':
            retval = self.transform_train(sample)

        elif self.split == 'val':
            retval = self.transform_val(sample)

        elif self.split == 'test':
            retval = self.transform_test(sample)

        if retval is None:
            raise Exception('Undefined split - should be either test/train/val')

        return retval


class ActiveCityscapesBase(CityscapesBase):

    def __init__(self, path, base_size, crop_size, split, init_set, overfit=False):

        super(ActiveCityscapesBase, self).__init__(path, base_size, crop_size, split, overfit)
        self.current_image_paths = []
        self.weakly_labeled_image_paths = []
        self.weakly_labeled_targets = {}

    def __len__(self):
        return len(self.current_image_paths) + len(self.weakly_labeled_image_paths)

    def _fix_list_multiple_of_batch_size(self, paths, batch_size):
        remainder = len(paths) % batch_size
        if remainder != 0:
            num_new_entries = batch_size - remainder
            new_entries = paths[:num_new_entries]
            paths.extend(new_entries)
        return paths

    def make_dataset_multiple_of_batchsize(self, batch_size):
        self.original_size_current = len(self.current_image_paths)
        self.original_size_weakly_labeled = len(self.weakly_labeled_image_paths)
        self.current_image_paths = self._fix_list_multiple_of_batch_size(self.current_image_paths, batch_size)
        self.weakly_labeled_image_paths = self._fix_list_multiple_of_batch_size(self.weakly_labeled_image_paths, batch_size)

    def reset_dataset(self):
        self.current_image_paths = self.current_image_paths[:self.original_size_current]
        self.weakly_labeled_image_paths = self.weakly_labeled_image_paths[:self.original_size_weakly_labeled]

    def get_fraction_of_labeled_data(self):
        return self.labeled_pixel_count / (len(self.image_paths) * self.crop_size * self.crop_size)

    def get_next_est_fraction_of_labeled_data(self, active_batch_size):
        return (self.labeled_pixel_count + active_batch_size * self.crop_size * self.crop_size) / (len(self.image_paths) * self.crop_size * self.crop_size)
