import argparse
import os
from tqdm import tqdm
import numpy as np
from dataloaders import make_dataloader

from models.sync_batchnorm.replicate import patch_replication_callback

from models.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import constants

class Evaluations:

	def __init__(self, args):
		self.args = args
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclass = make_dataloader(args.dataset, args.base_size, args.crop_size, args.batch_size, **kwargs)
		self.model = DeepLab(num_classes=self.nclass, backbone=args.backbone, output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
		self.evaluator = Evaluator(self.nclass)
		
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			patch_replication_callback(self.model)
			self.model = self.model.cuda()
	
		if args.use_balanced_weights:
			classes_weights_path = os.path.join(constants.DATASET_ROOT, args.dataset, 'class_weights.npy')
			if os.path.isfile(classes_weights_path):
				weight = np.load(classes_weights_path)
			else:
				weight = calculate_weights_labels(args.dataset, self.train_loader, self.nclass)
			weight = torch.from_numpy(weight.astype(np.float32))
		else:
			weight = None

		self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
		
		checkpoint = torch.load(args.resume)
		if args.cuda:
				self.model.module.load_state_dict(checkpoint['state_dict'])
		else:
			self.model.load_state_dict(checkpoint['state_dict'])
		print(f'=> loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')

		self.visualizations_folder = os.path.join(os.path.dirname(os.path.realpath(args.resume)), constants.VISUALIZATIONS_FOLDER)
		if not os.path.exists(self.visualizations_folder):
			os.makedirs(self.visualizations_folder)

	def evaluate(self):

		self.model.eval()
		self.evaluator.reset()

		tbar = tqdm(self.val_loader, desc='\r')
		test_loss = 0.0

		for i, sample in enumerate(self.test_loader):
			image, target = sample['image'], sample['label']

			if self.args.cuda:
				image, target = image.cuda(), target.cuda()

			with torch.no_grad():
				output = self.model(image)

			loss = self.criterion(output, target)
			test_loss += loss.item()
			tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
			pred = output.data.cpu().numpy()
			target = target.cpu().numpy()
			pred = np.argmax(pred, axis=1)
			self.evaluator.add_batch(target, pred)
			TensorboardSummary.visualize_images_to_folder(self.visualizations_folder, i, image.cpu().numpy(), target, pred, self.args.dataset)

		# Fast test during the training
		Acc = self.evaluator.Pixel_Accuracy()
		Acc_class = self.evaluator.Pixel_Accuracy_Class()
		mIoU = self.evaluator.Mean_Intersection_over_Union()
		FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
		print('Evaluation:')
		print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
		print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
		print('Loss: %.3f' % test_loss)


		

def main():

	parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Visualizations")
	parser.add_argument('--backbone', type=str, default='resnet',
						choices=['resnet', 'xception', 'drn', 'mobilenet'],
						help='backbone name (default: resnet)')
	parser.add_argument('--out-stride', type=int, default=16,
						help='network output stride (default: 8)')
	parser.add_argument('--dataset', type=str, default='cityscapes',
						choices=['pascal', 'coco', 'cityscapes'],
						help='dataset name (default: cityscapes)')
	parser.add_argument('--use-sbd', action='store_true', default=True,
						help='whether to use SBD dataset (default: True)')
	parser.add_argument('--workers', type=int, default=4,
						metavar='N', help='dataloader threads')
	parser.add_argument('--base-size', type=int, default=513,
						help='base image size')
	parser.add_argument('--crop-size', type=int, default=513,
						help='crop image size')
	parser.add_argument('--sync-bn', type=bool, default=None,
						help='whether to use sync bn (default: auto)')
	parser.add_argument('--freeze-bn', type=bool, default=False,
						help='whether to freeze bn parameters (default: False)')
	parser.add_argument('--loss-type', type=str, default='ce',
						choices=['ce', 'focal'],
						help='loss func type (default: ce)')
	parser.add_argument('--use-balanced-weights', action='store_true', default=False,
						help='whether to use balanced weights (default: False)')
	# training hyper params
	parser.add_argument('--batch-size', type=int, default=None,
						metavar='N', help='input batch size for \
								testing (default: auto)')
	# cuda, seed and logging
	parser.add_argument('--no-cuda', action='store_true', default=
						False, help='disables CUDA training')
	parser.add_argument('--gpu-ids', type=str, default='0',
						help='use which gpu to train, must be a \
						comma-separated list of integers only (default=0)')
	parser.add_argument('--resume', type=str, default=None,
						help='put the path to resuming file if needed')
	
	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

	if args.sync_bn is None:
		if args.cuda and len(args.gpu_ids) > 1:
			args.sync_bn = True
		else:
			args.sync_bn = False

	if args.batch_size is None:
		args.batch_size = 4 * len(args.gpu_ids)

	print(args)
	evaluations = Evaluations(args)
	evaluations.evaluate()
	

if __name__=='__main__':
	main()