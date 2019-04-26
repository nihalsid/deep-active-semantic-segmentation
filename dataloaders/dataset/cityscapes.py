import numpy as np
from PIL import Image
from torch.utils import data
import glob 
from pathlib import Path
import lmdb
import pickle
from dataloaders.dataset import cityscapes_base


class Cityscapes(cityscapes_base.CityscapesBase):

	
	def __init__(self, path, base_size, crop_size, split, overfit=False): 
		
		super(Cityscapes, self).__init__(path, split)

		self.path = path
		self.split = split
		self.crop_size = crop_size
		self.base_size = base_size
		self.overfit = overfit
		
		if overfit:
			self.image_paths = self.image_paths[:1]

		if len(self.image_paths) == 0:
			raise Exception("No images found in dataset directory")


	def __len__(self):
		return len(self.image_paths)


	def __getitem__(self, index):
		
		img_path = self.image_paths[index]
		
		loaded_npy = None
		with self.env.begin(write=False) as txn:
			loaded_npy = pickle.loads(txn.get(img_path))

		image = loaded_npy[:, :, 0:3]
		target = loaded_npy[:, :, 3]
	
		sample = {'image': Image.fromarray(image), 'label': Image.fromarray(target)}

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


	def replicate_training_set(self, factor):
		self.image_paths = self.image_paths * factor
		

	def reset_replicated_training_set(self):
		self.image_paths = list(set(self.image_paths))


	def set_paths(self, pathlist):
		self.image_paths = pathlist


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	import matplotlib.pyplot as plt
	from dataloaders.utils import map_segmentation_to_colors

	path = 'D:\\nihalsid\\DeeplabV3+\\datasets\\cityscapes'
	crop_size = 513
	base_size = 513
	split = 'test'
	
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