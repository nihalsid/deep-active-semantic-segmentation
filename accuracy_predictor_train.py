import argparse
import os
import numpy as np
from tqdm import tqdm
import math
import random

from dataloaders import make_dataloader
from models.sync_batchnorm.replicate import patch_replication_callback
import torch

from models.accuracy_predictor import DeepLabAccuracyPredictor
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver, ActiveSaver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from active_selection import get_active_selection_class
import constants
import sys
from utils.early_stop import EarlyStopChecker


class Trainer(object):

    def __init__(self, args, dataloaders):
        self.args = args
        self.train_loader, self.val_loader, self.test_loader, self.nclass = dataloaders

    def setup_saver_and_summary(self, num_current_labeled_samples, samples, experiment_group=None, regions=None):

        self.saver = ActiveSaver(self.args, num_current_labeled_samples, experiment_group=experiment_group)
        self.saver.save_experiment_config()
        self.saver.save_active_selections(samples, regions)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.num_current_labeled_samples = num_current_labeled_samples

    def initialize(self):

        args = self.args
        model = DeepLabAccuracyPredictor(num_classes=self.nclass, backbone=args.backbone, output_stride=args.out_stride,
                                         sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, mc_dropout=False)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10},
                        {'params': model.get_unet_params(), 'lr': args.lr}]

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(train_params)
        else:
            raise NotImplementedError

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

        self.criterion_deeplab = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion_unet = SegmentationLosses(weight=torch.FloatTensor(
            [args.weight_wrong_label_unet, 1 - args.weight_wrong_label_unet]), cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.deeplab_evaluator = Evaluator(self.nclass)
        self.unet_evaluator = Evaluator(2)

        if args.use_lr_scheduler:
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        else:
            self.scheduler = None

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0

    def training(self, epoch, w_dl, w_un):

        train_loss = 0.0
        train_loss_unet = 0.0
        train_loss_deeplab = 0.0

        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader, desc='\r')

        visualization_index = int(random.random() * len(self.train_loader))
        vis_img = None
        vis_tgt_dl = None
        vis_tgt_un = None
        vis_out_dl = None
        vis_out_un = None

        for i, sample in enumerate(tbar):
            image, deeplab_target = sample['image'], sample['label']

            if self.args.cuda:
                image, deeplab_target = image.cuda(), deeplab_target.cuda()
            if self.scheduler:
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.writer.add_scalar('train/learning_rate', self.scheduler.current_lr, i + num_img_tr * epoch)

            self.optimizer.zero_grad()
            deeplab_output, unet_output = self.model(image)
            unet_target = deeplab_output.argmax(1).squeeze() == deeplab_target.long()
            unet_target[deeplab_target == 255] = 255

            if i == visualization_index:
                vis_img = image.cpu()
                vis_tgt_dl = deeplab_target.cpu()
                vis_out_dl = deeplab_output.cpu()
                vis_tgt_un = unet_target.cpu()
                vis_out_un = unet_output.cpu()

            loss_deeplab = self.criterion_deeplab(deeplab_output, deeplab_target)
            loss_unet = self.criterion_unet(unet_output, unet_target)
            loss = w_dl * loss_deeplab + w_un * loss_unet
            loss.backward()
            self.optimizer.step()
            train_loss_deeplab += loss_deeplab.item()
            train_loss_unet += loss_unet.item()
            train_loss += loss.item()
            tbar.set_description('Train losses: %.2f(dl) + %.2f(un) = %.3f' %
                                 (train_loss_deeplab / (i + 1), train_loss_unet / (i + 1), train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter_dl', loss_deeplab.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/total_loss_iter_un', loss_unet.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.summary.create_single_visualization(self.writer, f'train/run_{self.num_current_labeled_samples:04d}', self.args.dataset, vis_img, vis_tgt_dl, vis_out_dl, vis_tgt_un, vis_out_un, epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/total_loss_epoch_dl', train_loss_unet, epoch)
        self.writer.add_scalar('train/total_loss_epoch_un', train_loss_deeplab, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f (DeepLab) + %.3f (UNet) = %.3f' % (train_loss_deeplab, train_loss_unet, train_loss))
        print('BestPred: %.3f' % self.best_pred)

        self.writer.add_scalar('train/w_dl', w_dl, i + num_img_tr * epoch)
        self.writer.add_scalar('train/w_un', w_un, i + num_img_tr * epoch)

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

    def validation(self, epoch, w_dl, w_un):

        self.model.eval()
        self.deeplab_evaluator.reset()
        self.unet_evaluator.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        test_loss_unet = 0.0
        test_loss_deeplab = 0.0

        visualization_index = int(random.random() * len(self.val_loader))
        vis_img = None
        vis_tgt_dl = None
        vis_tgt_un = None
        vis_out_dl = None
        vis_out_un = None

        for i, sample in enumerate(tbar):
            image, deeplab_target = sample['image'], sample['label']

            if self.args.cuda:
                image, deeplab_target = image.cuda(), deeplab_target.cuda()

            with torch.no_grad():
                deeplab_output, unet_output = self.model(image)

            unet_target = deeplab_output.argmax(1).squeeze() == deeplab_target.long()
            unet_target[deeplab_target == 255] = 255

            if i == visualization_index:
                vis_img = image.cpu()
                vis_tgt_dl = deeplab_target.cpu()
                vis_out_dl = deeplab_output.cpu()
                vis_tgt_un = unet_target.cpu()
                vis_out_un = unet_output.cpu()

            loss_deeplab = self.criterion_deeplab(deeplab_output, deeplab_target)
            loss_unet = self.criterion_unet(unet_output, unet_target)
            loss = w_dl * loss_deeplab + w_un * loss_unet

            test_loss += loss.item()
            test_loss_unet += loss_unet.item()
            test_loss_deeplab += loss_deeplab.item()

            tbar.set_description('Test losses: %.2f(dl) + %.2f(un) = %.3f' %
                                 (test_loss_deeplab / (i + 1), test_loss_unet / (i + 1), test_loss / (i + 1)))
            pred = deeplab_output.data.cpu().numpy()
            deeplab_target = deeplab_target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.deeplab_evaluator.add_batch(deeplab_target, pred)
            self.unet_evaluator.add_batch(unet_target.cpu().numpy(), np.argmax(unet_output.cpu().numpy(), axis=1))

        # Fast test during the training
        Acc = self.deeplab_evaluator.Pixel_Accuracy()
        Acc_class = self.deeplab_evaluator.Pixel_Accuracy_Class()
        mIoU = self.deeplab_evaluator.Mean_Intersection_over_Union()
        FWIoU = self.deeplab_evaluator.Frequency_Weighted_Intersection_over_Union()
        UNetAcc = self.unet_evaluator.Pixel_Accuracy()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/UNetAcc', UNetAcc, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, UNetAcc: {}".format(Acc, Acc_class, mIoU, FWIoU, UNetAcc))
        print('Loss: %.3f (DeepLab) + %.3f (UNet) = %.3f' % (test_loss_deeplab, test_loss_unet, test_loss))

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

        return test_loss, mIoU, Acc, Acc_class, FWIoU, [vis_img, vis_tgt_dl, vis_out_dl, vis_tgt_un, vis_out_un]


def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='active_cityscapes_image',
                        choices=['active_cityscapes_image', 'active_cityscapes_region'],
                        help='dataset name (default: active_cityscapes)')
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
    parser.add_argument('--workers', type=int, default=4,
                        help='num workers')
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
    parser.add_argument('--use-lr-scheduler', default=False, help='use learning rate scheduler', action='store_true')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
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
    parser.add_argument('--resume-selections', type=str, default=None,
                        help='resume selections file')
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
    parser.add_argument('--active-region-size', type=int, default=129, help='size of regions in case region dataset is used')
    parser.add_argument('--max-iterations', type=int, default=1000, help='maximum active selection iterations')
    parser.add_argument('--min-improvement', type=float, default=0.01, help='min improvement evaluation interval (default: 1)')
    parser.add_argument('--weight-unet', type=float, default=0.30, help='unet loss weight')
    parser.add_argument('--weight-wrong-label-unet', type=float, default=0.75, help='unet loss weight')
    parser.add_argument('--accuracy-selection', type=str, default='softmax', choices=['softmax', 'argmax'], help='selection based on soft or hard scores')
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
            'active_cityscapes': 50,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'active_cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print()
    print(args)

    w_dl = [1 - args.weight_unet] * args.epochs
    w_un = [args.weight_unet] * args.epochs

    if False:
        for i in range(args.epochs // 3, args.epochs):
            w_dl[i] = 0.75
            w_un[i] = 0.25

        for i in range(2 * args.epochs // 3, args.epochs):
            w_dl[i] = 0.5
            w_un[i] = 0.5

    kwargs = {'pin_memory': False, 'init_set': args.seed_set, 'memory_hog': args.memory_hog}
    dataloaders = make_dataloader(args.dataset, args.base_size, args.crop_size, args.batch_size, args.workers, args.overfit, **kwargs)

    training_set = dataloaders[0]
    dataloaders = dataloaders[1:]

    saver = Saver(args, remove_existing=False)
    saver.save_experiment_config()

    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    print()

    active_selector = get_active_selection_class('accuracy_labels', training_set.NUM_CLASSES, training_set.env, args.crop_size, args.batch_size)

    total_active_selection_iterations = min(len(training_set.image_paths) // args.active_batch_size - 1, args.max_iterations)

    if args.resume != 0 and args.resume_selections != None:
        seed_size = len(training_set)
        with open(os.path.join(saver.experiment_dir, args.resume_selections), "r") as fptr:
            paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']
        training_set.expand_training_set(paths[seed_size:])
        assert len(training_set) == (args.resume * args.active_batch_size + seed_size)

    assert args.eval_interval <= args.epochs and args.epochs % args.eval_interval == 0

    trainer = Trainer(args, dataloaders)
    trainer.initialize()

    for selection_iter in range(args.resume, total_active_selection_iterations):

        print(f'ActiveIteration-{selection_iter:03d}/{total_active_selection_iterations:03d}')

        fraction_of_data_labeled = round(training_set.get_fraction_of_labeled_data() * 100)

        if args.dataset == 'active_cityscapes_image':
            trainer.setup_saver_and_summary(fraction_of_data_labeled, training_set.current_image_paths)
        elif args.dataset == 'active_cityscapes_region':
            trainer.setup_saver_and_summary(fraction_of_data_labeled, training_set.current_image_paths, regions=[
                                            training_set.current_paths_to_regions_map[x] for x in training_set.current_image_paths])
        else:
            raise NotImplementedError

        len_dataset_before = len(training_set)
        training_set.make_dataset_multiple_of_batchsize(args.batch_size)
        print(f'\nExpanding training set with {len_dataset_before}  images to {len(training_set)} images')

        trainer.initialize()

        early_stop = EarlyStopChecker(patience=5, min_improvement=args.min_improvement)

        best_mIoU = 0
        best_Acc = 0
        best_Acc_class = 0
        best_FWIoU = 0

        for outer_epoch in range(args.epochs // args.eval_interval):
            train_loss = 0
            for inner_epoch in range(args.eval_interval):
                epoch = outer_epoch * args.eval_interval + inner_epoch
                train_loss += trainer.training(epoch, w_dl[epoch], w_un[epoch])
            test_loss, mIoU, Acc, Acc_class, FWIoU, visualizations = trainer.validation(epoch, w_dl[epoch], w_un[epoch])
            if mIoU > best_mIoU:
                best_mIoU = mIoU
            if Acc > best_Acc:
                best_Acc = Acc
            if Acc_class > best_Acc_class:
                best_Acc_class = Acc_class
            if FWIoU > best_FWIoU:
                best_FWIoU = FWIoU
            # check for early stopping
            if early_stop(mIoU):
                print(f'Early stopping triggered after {epoch} epochs')
                break

        training_set.reset_dataset()

        writer.add_scalar('active_loop/train_loss', train_loss / len(training_set), fraction_of_data_labeled)
        writer.add_scalar('active_loop/val_loss', test_loss, fraction_of_data_labeled)
        writer.add_scalar('active_loop/mIoU', best_mIoU, fraction_of_data_labeled)
        writer.add_scalar('active_loop/Acc', best_Acc, fraction_of_data_labeled)
        writer.add_scalar('active_loop/Acc_class', best_Acc_class, fraction_of_data_labeled)
        writer.add_scalar('active_loop/fwIoU', best_FWIoU, fraction_of_data_labeled)

        summary.create_single_visualization(writer, f'active_loop', args.dataset, visualizations[0], visualizations[1], visualizations[
            2], visualizations[3], visualizations[4], len(training_set.current_image_paths))

        trainer.writer.close()
        trainer.model.eval()

        print('Estimating accuracies..')

        selected_images = active_selector.get_least_accurate_samples(
            trainer.model, training_set.remaining_image_paths, args.active_batch_size, args.accuracy_selection)
        training_set.expand_training_set(selected_images)
        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()
