from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils import data
import lmdb
import pickle
import os
from enum import Enum


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

        if overfit:
            self.image_paths = self.image_paths[:1]

        if len(self.image_paths) == 0:
            raise Exception("No images found in dataset directory")

    def transform_train(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
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


class Mode(Enum):
    ALL_BATCHES = 0
    LAST_ADDED_BATCH = 1


class ActiveCityscapesBase(CityscapesBase):

    def __init__(self, path, base_size, crop_size, split, init_set, overfit=False):

        super(ActiveCityscapesBase, self).__init__(path, base_size, crop_size, split, overfit)
        self.current_image_paths = []
        self.last_added_image_paths = []
        self.mode = Mode.ALL_BATCHES

    def set_mode_all(self):
        self.mode = Mode.ALL_BATCHES

    def set_mode_last(self):
        self.mode = Mode.LAST_ADDED_BATCH

    def __len__(self):
        if self.mode == Mode.ALL_BATCHES:
            return len(self.current_image_paths)
        else:
            return len(self.last_added_image_paths)

    def replicate_training_set(self, factor):
        self.current_image_paths = self.current_image_paths * factor
        self.last_added_image_paths = self.last_added_image_paths * factor

    def reset_replicated_training_set(self):
        self.current_image_paths = list(set(self.current_image_paths))
        self.last_added_image_paths = list(set(self.last_added_image_paths))
