import argparse
import os
import numpy as np
from tqdm import tqdm

from dataloaders import make_dataloader
from models.sync_batchnorm.replicate import patch_replication_callback

from models.deeplab import *
from models.enet import ENet
from models.fastscnn import FastSCNN
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import PassiveSaver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import random

import constants

import sys


class Trainer(object):

    def __init__(self, args):
        self.args = args

        self.saver = PassiveSaver(args)
        self.saver.save_experiment_config()

        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'pin_memory': False, 'memory_hog': args.memory_hog}
        self.train_set, self.train_loader, self.val_loader, self.test_loader, self.nclass = make_dataloader(
            args.dataset, args.base_size, args.crop_size, args.batch_size, args.workers, args.overfit, **kwargs)

        self.train_set.make_dataset_multiple_of_batchsize(args.batch_size)

        if args.architecture == 'deeplab':
            print('Using Deeplab')
            model = DeepLab(num_classes=self.nclass, backbone=args.backbone, output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        elif args.architecture == 'enet':
            print('Using ENet')
            model = ENet(num_classes=self.nclass, encoder_relu=True, decoder_relu=True)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]
        elif args.architecture == 'fastscnn':
            print('Using FastSCNN')
            model = FastSCNN(3, self.nclass)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(train_params, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        if args.use_balanced_weights:
            weight = calculate_weights_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.evaluator = Evaluator(self.nclass)

        if args.use_lr_scheduler:
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        else:
            self.scheduler = None

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at {args.resume}")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f'=> loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')

    def training(self, epoch):

        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc='\r')

        visualization_index = int(random.random() * len(self.val_loader))
        vis_img = None
        vis_tgt = None
        vis_out = None

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            if self.scheduler:
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.writer.add_scalar('train/learning_rate', self.scheduler.current_lr, i + num_img_tr * epoch)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if i == visualization_index:
                vis_img = image
                vis_tgt = target
                vis_out = output

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.summary.visualize_image(self.writer, self.args.dataset, vis_img, vis_tgt, vis_out, epoch, prefix='train')

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
        self.summary.visualize_image(self.writer, self.args.dataset, vis_img, vis_tgt, vis_out, epoch)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: cityscapes)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                                            help='whether to use SBD dataset (default: False)')
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
                        help='whether to use balanced weights (default: True)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--use-lr-scheduler', default=False, help='use learning rate scheduler', action='store_true')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
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
    parser.add_argument('--architecture', type=str, default='deeplab', choices=['deeplab', 'enet', 'fastscnn'])
    parser.add_argument('--memory-hog', action='store_true', default=False, help='memory_hog mode')

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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
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
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
