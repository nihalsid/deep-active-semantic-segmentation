import argparse
import os
import sys
import numpy as np
import argparse
import math

from tqdm import tqdm
from dataloaders import make_dataloader
from active_train import Trainer
import torch
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.early_stop import EarlyStopChecker


def main():

	parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

	parser.add_argument('--backbone', type=str, default='resnet',
						choices=['resnet', 'xception', 'drn', 'mobilenet'],
						help='backbone name (default: resnet)')
	parser.add_argument('--out-stride', type=int, default=16,
						help='network output stride (default: 8)')
	parser.add_argument('--dataset', type=str, default='cityscapes',
						choices=['pascal', 'coco', 'cityscapes'],
						help='dataset name (default: cityscapes)')
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
	# training hyper params
	parser.add_argument('--epochs', type=int, default=None, metavar='N',
						help='number of epochs to train (default: auto)')
	parser.add_argument('--start_epoch', type=int, default=0,
						metavar='N', help='start epochs (default:0)')
	parser.add_argument('--batch-size', type=int, default=None,
						metavar='N', help='input batch size for \
								training (default: auto)')
	parser.add_argument('--use-balanced-weights', action='store_true', default=False,
						help='whether to use balanced weights (default: False)')
	# optimizer params
	parser.add_argument('--lr', type=float, default=None, metavar='LR',
						help='learning rate (default: auto)')
	parser.add_argument('--lr-scheduler', type=str, default='poly',
						choices=['poly', 'step', 'cos'],
						help='lr scheduler mode: (default: poly)')
	parser.add_argument('--momentum', type=float, default=0.9,
						metavar='M', help='momentum (default: 0.9)')
	parser.add_argument('--weight-decay', type=float, default=5e-4,
						metavar='M', help='w-decay (default: 5e-4)')
	parser.add_argument('--nesterov', action='store_true', default=False,
						help='whether use nesterov (default: False)')
	# cuda, seed and logging
	parser.add_argument('--no-cuda', action='store_true', default=
						False, help='disables CUDA training')
	parser.add_argument('--gpu-ids', type=str, default='0',
						help='use which gpu to train, must be a \
						comma-separated list of integers only (default=0)')
	# checking point
	parser.add_argument('--resume', type=int, default=0,
						help='iteration to resume from')
	parser.add_argument('--checkname', type=str, default=None,
						help='set the checkpoint name')
	# evaluation option
	parser.add_argument('--eval-interval', type=int, default=1,
						help='evaluation interval (default: 1)')
	parser.add_argument('--no-val', action='store_true', default=False,
						help='skip validation during training')

	parser.add_argument('--active-experiment-dir', 
						help='directory containing the images selected during various active learning selection iterations')
	parser.add_argument('--active-evaluation-skips', type=int, default=10,
						help='evaluation interval (default: 1)')
	parser.add_argument('--min-improvement', type=float, default=0.01,
						help='evaluation interval (default: 1)')
	parser.add_argument('--experiment-group', type=str, default='cityscape_active_evals',
						help='root folder for saving experiments')

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

	mc_dropout = False


	active_iteration_dirs = [x for x in os.listdir(args.active_experiment_dir) if x.startswith('run_')]
	active_iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))

	active_selections = []

	for i in range(len(active_iteration_dirs)): 
		with open(os.path.join(args.active_experiment_dir, active_iteration_dirs[i], 'selections.txt'), "r") as fptr:
			paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
			active_selections.append(paths)

	print()
	print(args)

	kwargs = {'pin_memory': True}
	
	dataloaders = make_dataloader(args.dataset, args.base_size, args.crop_size, args.batch_size, False, **kwargs)
	
	training_set = dataloaders[0]
	dataloaders = dataloaders[1:]

	saver = Saver(args, remove_existing=False, experiment_group=args.experiment_group)
	saver.save_experiment_config()

	summary = TensorboardSummary(saver.experiment_dir)
	writer = summary.create_summary()
	
	trainer = Trainer(args, dataloaders, mc_dropout)

	print()

	assert (args.eval_interval <= args.epochs)
	effective_epochs = math.ceil(args.epochs / args.eval_interval)

	for selection_iter in range(args.resume, len(active_selections), args.active_evaluation_skips):

		training_set.set_paths(active_selections[selection_iter])

		print(f'SelectionIter[{selection_iter:03d}]: #samples = {len(training_set):05d}')

		trainer.setup_saver_and_summary(len(training_set), training_set.image_paths, experiment_group=args.experiment_group)
		
		trainer.initialize()
		
		print(f'Expanding training set with {len(training_set)} images to {len(training_set) * args.eval_interval} images')

		training_set.replicate_training_set(args.eval_interval)
		
		early_stop = EarlyStopChecker(patience=2, min_improvement=args.min_improvement)

		for epoch in range(effective_epochs):
			train_loss = trainer.training(epoch)
			test_loss, mIoU, Acc, Acc_class, FWIoU, visualizations = trainer.validation(epoch)

			#check for early stopping 
			if early_stop(mIoU):
				print (f'Early stopping triggered after {epoch * args.eval_interval} epochs')
				break

		training_set.reset_replicated_training_set()

		writer.add_scalar('active_loop/train_loss', train_loss / len(training_set), len(training_set))
		writer.add_scalar('active_loop/val_loss', test_loss, len(training_set))
		writer.add_scalar('active_loop/mIoU', mIoU, len(training_set))
		writer.add_scalar('active_loop/Acc', Acc, len(training_set))
		writer.add_scalar('active_loop/Acc_class', Acc_class, len(training_set))
		writer.add_scalar('active_loop/fwIoU', FWIoU, len(training_set))

		summary.visualize_image(writer, args.dataset, visualizations[0], visualizations[1], visualizations[2], len(training_set))

		trainer.writer.close()
		

if __name__ == "__main__":
	main()