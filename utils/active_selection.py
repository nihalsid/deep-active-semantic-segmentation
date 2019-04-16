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




class ActiveSelectionMCDropout:

	def __init__(self, dataset_classes, crop_size, num_workers, batch_size, cuda):
		self.dataset_classes = dataset_classes
		self.crop_size = crop_size
		self.num_workers = num_workers
		self.cuda = cuda
		self.batch_size = batch_size


	def get_random_uncertainity(self, images):
		scores = []
		for i in range(len(images)):
			scores.append(random.random())
		return scores


	def _get_uncertainity_for_batch(self, model, image_batch):
		outputs = np.zeros([image_batch.shape[0], constants.MC_STEPS, self.dataset_classes, image_batch.shape[2], image_batch.shape[3]])

		with torch.no_grad():
			for step in range(constants.MC_STEPS):
				outputs[:, step, :,  :, :] = model(image_batch).cpu().numpy().squeeze()

		variances = []

		for i in range(image_batch.shape[0]):
			
			# visualize for debugging
			#prediction = np.argmax(outputs[i, :, :, :, :].squeeze().mean(axis=0), axis=0)
			#self._visualize_variance(image_batch[i, :, :, :].cpu().numpy().squeeze(), np.sum(np.var(outputs[i, :, :, :, :].squeeze(), axis=0), axis=0) / (constants.MC_STEPS * self.dataset_classes), prediction)
			
			variances.append(np.sum(np.var(outputs[i, :, :, :, :].squeeze(), axis=0)) / (constants.MC_STEPS * self.dataset_classes * image_batch.shape[2] * image_batch.shape[3]))

		return variances


	def get_uncertainity_for_images(self, model, images):
		
		def turn_on_dropout(m):
			if type(m) == nn.Dropout2d:
				m.train()
		model.apply(turn_on_dropout)

		loader = DataLoader(active_cityscapes.PathsDataset(images, self.crop_size), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
		variances = []
		for image_batch in tqdm(loader):
			if self.cuda:
				image_batch = image_batch.cuda()
			variances.extend(self._get_uncertainity_for_batch(model, image_batch))
		
		model.eval()

		return variances
	

	def _visualize_variance(self, image_normalized, variance_map, prediction):
		image_unnormalized = ((np.transpose(image_normalized, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
		prediction_mapped = map_segmentation_to_colors(prediction.astype(np.uint8), 'cityscapes')
		norm = matplotlib.colors.Normalize(vmin = np.min(variance_map), vmax = np.max(variance_map), clip = False)
		plt.figure()
		plt.title('display')
		plt.subplot(1, 3, 1)
		plt.imshow(image_unnormalized)
		plt.subplot(1, 3, 2)
		plt.imshow(prediction_mapped)
		plt.subplot(1, 3, 3)
		plt.imshow(variance_map, norm=norm, cmap='gray')
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
		'cuda': True,
		'num_workers': 4,
		'seed_set': 'set_1.txt',
		'batch_size': 4
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

	active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, args.crop_size, args.num_workers, args.batch_size, args.cuda)
	print(active_selector.get_uncertainity_for_images(model, train_set.current_image_paths))
