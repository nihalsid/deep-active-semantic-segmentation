import os
import numpy as np
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils import data
import glob 
from pathlib import Path


class Cityscapes(data.Dataset):
	
	NUM_CLASSES = 19
	
	def __init__(self, path, base_size, crop_size, split, overfit=False): 
		
		self.path = path
		self.split = split
		self.crop_size = crop_size
		self.base_size = base_size
		self.overfit = overfit

		self.images_base = os.path.join(self.path, 'leftImg8bit', "train" if self.overfit else self.split)
		self.labels_base = os.path.join(self.path, 'gtFine_trainvaltest', 'gtFine', "train" if self.overfit else self.split)

		self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
		self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
		self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
		self.ignore_index = 255
		self.class_map = dict(zip(self.valid_classes + self.void_classes, list(range(self.NUM_CLASSES)) + [self.ignore_index] * len(self.void_classes)))
		
		self.image_paths = glob.glob(os.path.join(self.images_base, '**', '*.png'), recursive=True)
		
		if overfit:
			self.image_paths = self.image_paths[:1]

		if len(self.image_paths) == 0:
			raise Exception("No images found in dataset directory")


	def __len__(self):
		return len(self.image_paths)


	def __getitem__(self, index):

		img_path = self.image_paths[index]
		lbl_path = os.path.join(self.labels_base, Path(img_path).parts[-2], f'{os.path.basename(img_path)[:-15]}gtFine_labelIds.png')

		image = Image.open(img_path).convert('RGB')
		target = np.array(Image.open(lbl_path), dtype=np.uint8)

		mapper = np.vectorize(lambda l: self.class_map[l])
		target[:, :] = mapper(target)
			
		sample = {'image': image, 'label': Image.fromarray(target)}

		if self.overfit:
			return self.transform_test(sample)

		if self.split == 'train':
			return self.transform_train(sample)

		elif self.split == 'val':
			return self.transform_val(sample)

		elif self.split == 'test':
			return self.transform_test(sample)

		raise Exception('Undefined split - should be either test/train/val')

	
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


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	import matplotlib.pyplot as plt
	from dataloaders.utils import map_segmentation_to_colors

	path = 'D:\\nihalsid\\DeeplabV3+\\datasets\\cityscapes'
	crop_size = 513
	base_size = 513
	split = 'test'
	
	cityscapes_train = Cityscapes(path, base_size, crop_size, split)
	dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

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