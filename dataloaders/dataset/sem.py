import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data.dataset import Dataset
import constants
import os


class SEMData(Dataset):

    def __init__(self, crop_size, split):
        self.mask_paths = glob.glob(os.path.join(constants.DATASET_ROOT,  f'sem/{split}/masks/*'))
        self.image_paths = glob.glob(os.path.join(constants.DATASET_ROOT, f'sem/{split}/images/*'))
        self.split = split
        self.crop_size = crop_size

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]))
        mask = np.array(Image.open(self.mask_paths[index])) / 255.0
        sample = {'image': Image.fromarray(image), 'label': Image.fromarray(mask)}
        return self.get_transformed_sample(sample)

    def __len__(self):
        return len(self.image_paths)

    def transform_train(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=[0.4911], std=[0.1658]),
            tr.ToTensor()
        ])
        transformed = composed_transforms(sample)
        transformed['image'] = transformed['image'].unsqueeze(0)
        return transformed

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=[0.4911], std=[0.1658]),
            tr.ToTensor()
        ])

        transformed = composed_transforms(sample)
        transformed['image'] = transformed['image'].unsqueeze(0)
        return transformed

    def get_transformed_sample(self, sample):

        retval = None

        if self.split == 'train':
            retval = self.transform_train(sample)
        elif self.split == 'val':
            retval = self.transform_val(sample)

        if retval is None:
            raise Exception('Undefined split - should be either test/train/val')

        return retval

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    crop_size = 512
    split = 'val'

    dataset = SEMData(crop_size, split)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader, 0):
        for j in range(sample['image'].size()[0]):
            image = sample['image'].numpy()[j]
            gt = sample['label'].numpy()[j]
            image_unnormalized = ((image * 0.1658 + 0.4911) * 255).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized.squeeze())
            plt.subplot(212)
            plt.imshow(gt.squeeze())

        if i == 1:
            break

    plt.show(block=True)
