from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils import data
import lmdb
import pickle
import os


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
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
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
