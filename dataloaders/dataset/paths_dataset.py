from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils import data
import pickle
from PIL import Image


class PathsDataset(data.Dataset):

    def __init__(self, env, paths, crop_size, include_labels=False):

        self.env = env
        self.paths = paths
        self.crop_size = crop_size
        self.include_labels = include_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path = self.paths[index]
        loaded_npy = None

        with self.env.begin(write=False) as txn:
            loaded_npy = pickle.loads(txn.get(img_path))

        image = loaded_npy[:, :, 0:3]
        target = loaded_npy[:, :, 3]

        if self.include_labels:
            composed_tr = transforms.Compose([
                tr.FixScaleCrop(crop_size=self.crop_size),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                tr.ToTensor()
            ])
            return composed_tr({'image': image, 'label': target})
        else:
            composed_tr = transforms.Compose([
                tr.FixScaleCropImageOnly(crop_size=self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            return composed_tr(image)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataloaders.dataset import active_cityscapes
    from dataloaders.utils import map_segmentation_to_colors
    import os
    import numpy as np
    import constants

    path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    crop_size = 513
    base_size = 513
    split = 'train'

    cityscapes_train = active_cityscapes.ActiveCityscapesImage(path, base_size, crop_size, split, 'set_0.txt')
    loader = DataLoader(PathsDataset(cityscapes_train.env, cityscapes_train.current_image_paths,
                                     crop_size, include_labels=True), batch_size=2, shuffle=False, num_workers=0)

    for i, sample in enumerate(loader, 0):
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
