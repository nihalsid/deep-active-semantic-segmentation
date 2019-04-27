import argparse
import os
import numpy as np
from tqdm import tqdm
import math
import random

from dataloaders import make_dataloader
from models.sync_batchnorm.replicate import patch_replication_callback

from models.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver, ActiveSaver
from utils.summaries import TensorboardSummary
from utils.active_selection import ActiveSelectionMCDropout
from utils.metrics import Evaluator
import constants
import sys
from utils.early_stop import EarlyStopChecker


class Trainer(object):

    def __init__(self, args, dataloaders, mc_dropout):
        self.args = args
        self.mc_dropout = mc_dropout
        self.train_loader, self.val_loader, self.test_loader, self.nclass = dataloaders

    def setup_saver_and_summary(self, num_current_labeled_samples, samples, experiment_group=None, regions=None):

        self.saver = ActiveSaver(self.args, num_current_labeled_samples, experiment_group=experiment_group)
        self.saver.save_experiment_config()
        self.saver.save_active_selections(samples, regions)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

    def initialize(self):

        args = self.args

        model = DeepLab(num_classes=self.nclass, backbone=args.backbone, output_stride=args.out_stride,
                        sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, mc_dropout=self.mc_dropout)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.use_balanced_weights:
            dataset_folder = args.dataset
            if args.dataset == 'active_cityscapes':
                dataset_folder = 'cityscapes'
            classes_weights_path = os.path.join(constants.DATASET_ROOT, dataset_folder, 'class_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weights_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.evaluator = Evaluator(self.nclass)

        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0

    def training(self, epoch):

        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('BestPred: %.3f' % self.best_pred)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        return train_loss

    def validation(self, epoch):

        self.model.eval()
        self.evaluator.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        visualization_index = int(random.random() * len(self.val_loader))
        vis_img = None
        vis_tgt = None
        vis_out = None

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            if i == visualization_index:
                vis_img = image
                vis_tgt = target
                vis_out = output

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        is_best = False
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        # save every validation model (overwrites)
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)

        return test_loss, mIoU, Acc, Acc_class, FWIoU, [vis_img, vis_tgt, vis_out]


def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='active_cityscapes_image',
                        choices=['active_cityscapes_image', 'active_cityscapes_region'],
                        help='dataset name (default: active_cityscapes)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                                            help='whether to use SBD dataset (default: False)')
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
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
								testing (default: auto)')
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
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
						comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=int, default=0,
                        help='iteration to resume from')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--overfit', action='store_true', default=False,
                                            help='overfit to one sample')
    parser.add_argument('--seed_set', action='store_true', default='set_0.txt',
                        help='initial labeled set')
    parser.add_argument('--active-batch-size', type=int, default=50,
                        help='batch size queried from oracle')
    parser.add_argument('--active-train-mode', type=str, default='scratch',
                        help='whether to reset model after each active loop or train only on new data', choices=['last', 'mix', 'scratch'])
    parser.add_argument('--active-selection-mode', type=str, default='random', choices=['random', 'variance'], help='method to select new samples')
    parser.add_argument('--active-region-size', type=int, default=127, help='size of regions in case region dataset is used')
    parser.add_argument('--max-iterations', type=int, default=1000, help='maximum active selection iterations')
    parser.add_argument('--min-improvement', type=float, default=0.01, help='evaluation interval (default: 1)')

    args = parser.parse_args()

    if args.active_selection_mode == "random":
        assert args.dataset == 'active_cityscapes_image', "For random mode only images supported, not regions"

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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'active_cityscapes': 50,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'active_cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)

    mc_dropout = True

    print()
    print(args)
    torch.manual_seed(args.seed)

    kwargs = {'pin_memory': True, 'init_set': args.seed_set}
    dataloaders = make_dataloader(args.dataset, args.base_size, args.crop_size, args.batch_size, args.overfit, **kwargs)

    training_set = dataloaders[0]
    dataloaders = dataloaders[1:]

    saver = Saver(args, remove_existing=False)
    saver.save_experiment_config()

    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    print()

    trainer = Trainer(args, dataloaders, mc_dropout)
    trainer.initialize()

    active_selector = ActiveSelectionMCDropout(training_set.NUM_CLASSES, training_set.env, args.crop_size, args.batch_size)

    total_active_selection_iterations = min(len(training_set.image_paths) // args.active_batch_size - 1, args.max_iterations)

    for i in range(args.resume):
        training_set.expand_training_set(active_selection.get_random_uncertainity(training_set.remaining_image_paths), args.active_batch_size)

    assert (args.eval_interval <= args.epochs)
    effective_epochs = math.ceil(args.epochs / args.eval_interval)

    if args.active_train_mode == 'last':
        training_set.set_mode_last()
    if args.active_train_mode == 'scratch':
        training_set.set_mode_all()
    else:
        raise NotImplementedError

    for selection_iter in range(args.resume, total_active_selection_iterations):

        print(f'ActiveIteration-{selection_iter:03d}/{total_active_selection_iterations:03d}')

        if args.dataset == 'active_cityscapes_image':
            trainer.setup_saver_and_summary(len(training_set.current_image_paths), training_set.current_image_paths)
        elif args.dataset == 'active_cityscapes_region':
            trainer.setup_saver_and_summary(len(training_set.current_image_paths), training_set.current_image_paths, regions=[
                                            training_set.current_paths_to_regions_map[x] for x in training_set.current_image_paths])
        else:
            raise NotImplementedError

        if args.active_train_mode == 'scratch':
            trainer.initialize()

        print(f'\nExpanding training set with {len(training_set)} images to {len(training_set) * args.eval_interval} images')
        training_set.replicate_training_set(args.eval_interval)

        early_stop = EarlyStopChecker(patience=2, min_improvement=args.min_improvement)

        for epoch in range(effective_epochs):
            train_loss = trainer.training(epoch)
            test_loss, mIoU, Acc, Acc_class, FWIoU, visualizations = trainer.validation(epoch)

        training_set.reset_replicated_training_set()

        fraction_of_data_labeled = round(training_set.get_fraction_of_labeled_data())

        writer.add_scalar('active_loop/train_loss', train_loss / len(training_set), fraction_of_data_labeled)
        writer.add_scalar('active_loop/val_loss', test_loss, fraction_of_data_labeled)
        writer.add_scalar('active_loop/mIoU', mIoU, fraction_of_data_labeled)
        writer.add_scalar('active_loop/Acc', Acc, fraction_of_data_labeled)
        writer.add_scalar('active_loop/Acc_class', Acc_class, fraction_of_data_labeled)
        writer.add_scalar('active_loop/fwIoU', FWIoU, fraction_of_data_labeled)

        summary.visualize_image(writer, args.dataset, visualizations[0], visualizations[1], visualizations[2], len(training_set.current_image_paths))

        trainer.writer.close()

        if args.active_selection_mode == 'random':
            training_set.expand_training_set(active_selector.get_random_uncertainity(training_set.remaining_image_paths), args.active_batch_size)
        elif args.active_selection_mode == 'variance':
            trainer.model.eval()
            if args.dataset == 'active_cityscapes_image':
                training_set.expand_training_set(active_selector.get_vote_entropy_for_images(
                    trainer.model, training_set.remaining_image_paths), args.active_batch_size)
            elif args.dataset == 'active_cityscapes_region':
                regions, counts = active_selector.create_region_maps(
                    trainer.model, training_set.image_paths, training_set.get_existing_region_maps(), args.active_region_size, args.active_batch_size)
                print(f'Got {counts}/{math.ceil(args.active_batch_size * args.crop_size * args.crop_size / (args.active_region_size * args.active_region_size))} regions')
                training_set.expand_training_set(regions, counts * args.active_region_size * args.active_region_size)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    writer.close()

if __name__ == "__main__":
    main()
