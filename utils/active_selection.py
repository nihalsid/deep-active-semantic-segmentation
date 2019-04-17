import constants
import numpy as np
import random
from dataloaders.dataset import active_cityscapes
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib
import matplotlib.pyplot as plt
from dataloaders.utils import map_segmentation_to_colors
from scipy import stats


class ActiveSelectionMCDropout:

	def __init__(self, dataset_num_classes, dataset_lmdb_env, crop_size, batch_size):
		self.dataset_num_classes = dataset_num_classes
		self.crop_size = crop_size
		self.batch_size = batch_size
		self.env = dataset_lmdb_env


	def get_random_uncertainity(self, images):
		scores = []
		for i in range(len(images)):
			scores.append(random.random())
		return scores


	def _get_uncertainity_for_batch(self, model, image_batch):
		
		outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
		with torch.no_grad():
			for step in range(constants.MC_STEPS):
				outputs[:, step, :, :] = torch.argmax(model(image_batch), dim=1)

		entropies = []

		for i in range(image_batch.shape[0]):
			entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)
			
			for c in range(self.dataset_num_classes):
				p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
				entropy_map = entropy_map - (p * torch.log2(p + 1e-12))
			
			# visualize for debugging
			#prediction = stats.mode(outputs[i, :, :, :].cpu().numpy(), axis=0)[0].squeeze()
			#self._visualize_entropy(image_batch[i, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
			entropies.append(torch.sum(entropy_map).cpu().item() / (image_batch.shape[2] * image_batch.shape[3]))
		
		return entropies


	def get_uncertainity_for_images(self, model, images):
		
		def turn_on_dropout(m):
			if type(m) == nn.Dropout2d:
				m.train()
		model.apply(turn_on_dropout)

		loader = DataLoader(active_cityscapes.PathsDataset(self.env, images, self.crop_size), batch_size=self.batch_size, shuffle=False, num_workers=0)

		entropies = []
		for image_batch in tqdm(loader):
			image_batch = image_batch.cuda()
			entropies.extend(self._get_uncertainity_for_batch(model, image_batch))
		
		model.eval()

		return entropies
	

	def _visualize_entropy(self, image_normalized, entropy_map, prediction):
		image_unnormalized = ((np.transpose(image_normalized, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
		prediction_mapped = map_segmentation_to_colors(prediction.astype(np.uint8), 'cityscapes')
		norm = matplotlib.colors.Normalize(vmin = np.min(entropy_map), vmax = np.max(entropy_map), clip = False)
		plt.figure()
		plt.title('display')
		plt.subplot(1, 3, 1)
		plt.imshow(image_unnormalized)
		plt.subplot(1, 3, 2)
		plt.imshow(prediction_mapped)
		plt.subplot(1, 3, 3)
		plt.imshow(entropy_map, norm=norm, cmap='gray')
		plt.show(block=True)


if __name__ == '__main__':

	def validation(model, val_loader, args):

		from utils.loss import SegmentationLosses
		from utils.metrics import Evaluator
		
		evaluator = Evaluator(19)
		evaluator.reset()

		tbar = tqdm(val_loader, desc='\r')
		test_loss = 0.0
		criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')

		for i, sample in enumerate(tbar):
			image, target = sample['image'], sample['label']
			
			if args.cuda:
				image, target = image.cuda(), target.cuda()
			
			with torch.no_grad():
				output = model(image)

			loss = criterion(output, target)
			test_loss += loss.item()
			tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
			pred = output.data.cpu().numpy()
			target = target.cpu().numpy()
			pred = np.argmax(pred, axis=1)
			evaluator.add_batch(target, pred)

		# Fast test during the training
		Acc = evaluator.Pixel_Accuracy()
		Acc_class = evaluator.Pixel_Accuracy_Class()
		mIoU = evaluator.Mean_Intersection_over_Union()
		FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
		print('Validation:')
		print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
		print('Loss: %.3f' % test_loss)

	class dotdict(dict):
	    """dot.notation access to dictionary attributes"""
	    __getattr__ = dict.get
	    __setattr__ = dict.__setitem__
	    __delattr__ = dict.__delitem__

	args = {
		'base_size': 513,
		'crop_size': 513,
		'seed_set': '',
		'seed_set': 'set_0.txt',
		'batch_size': 16
	}

	args = dotdict(args)
	dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
	train_set = active_cityscapes.ActiveCityscapes(path=dataset_path, base_size=args.base_size, crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
	val_set = active_cityscapes.ActiveCityscapes(path=dataset_path, base_size=args.base_size, crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
	model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)
	
	model = torch.nn.DataParallel(model, device_ids=[0])
	patch_replication_callback(model)
	model = model.cuda()
	
	checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes', 'al_0-random-overfit50-bs_50-deeplab-mobilenet-bs_4-513x513', 'run_0050', 'best.pth.tar'))
	model.module.load_state_dict(checkpoint['state_dict'])
	
	model.eval()
	
	# ensure that the loaded model is not crap
	# validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

	active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
	print(active_selector.get_uncertainity_for_images(model, train_set.current_image_paths))
