import constants
import numpy as np
import random
from dataloaders.dataset import paths_dataset
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from dataloaders.utils import map_segmentation_to_colors
from scipy import stats


class ActiveSelectionMCDropout:

    def __init__(self, dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        self.dataset_num_classes = dataset_num_classes
        self.crop_size = crop_size
        self.dataloader_batch_size = dataloader_batch_size
        self.env = dataset_lmdb_env

    def get_random_uncertainity(self, images):
        scores = []
        for i in range(len(images)):
            scores.append(random.random())
        return scores

    def _get_vote_entropy_for_batch(self, model, image_batch):

        outputs = torch.cuda.FloatTensor(image_batch.shape[0], constants.MC_STEPS, image_batch.shape[2], image_batch.shape[3])
        with torch.no_grad():
            for step in range(constants.MC_STEPS):
                outputs[:, step, :, :] = torch.argmax(model(image_batch), dim=1)

        entropy_maps = []

        for i in range(image_batch.shape[0]):
            entropy_map = torch.cuda.FloatTensor(image_batch.shape[2], image_batch.shape[3]).fill_(0)

            for c in range(self.dataset_num_classes):
                p = torch.sum(outputs[i, :, :, :] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS
                entropy_map = entropy_map - (p * torch.log2(p + 1e-12))

            # visualize for debugging
            # prediction = stats.mode(outputs[i, :, :, :].cpu().numpy(), axis=0)[0].squeeze()
            # self._visualize_entropy(image_batch[i, :, :, :].cpu().numpy(), entropy_map.cpu().numpy(), prediction)
            entropy_maps.append(entropy_map)

        return entropy_maps

    @staticmethod
    def square_nms(score_maps, region_size, max_selection_count):

        score_maps[score_maps < 0.01] = 0
        ones_tensor = torch.cuda.FloatTensor(score_maps.shape[1], score_maps.shape[2]).fill_(0)
        selected_indices = [[] for x in range(score_maps.shape[0])]
        selection_count = 0

        while score_maps.nonzero().size(0) != 0 and selection_count < max_selection_count:

            argmax = score_maps.view(-1).argmax()
            i, r, c = argmax // (score_maps.shape[1] * score_maps.shape[2]), (argmax //
                                                                              score_maps.shape[2]) % score_maps.shape[1], argmax % score_maps.shape[2]

            selected_indices[i.item()].append((r.item(), c.item(), region_size, region_size))
            selection_count += 1
            zero_out_mask = ones_tensor != 0
            r0 = max(0, r - region_size)
            c0 = max(0, c - region_size)
            r1 = min(score_maps.shape[1], r + region_size)
            c1 = min(score_maps.shape[2], c + region_size)
            zero_out_mask[r0:r1, c0:c1] = 1
            score_maps[i, zero_out_mask] = 0

        return selected_indices, selection_count

    @staticmethod
    def suppress_labeled_entropy(entropy_map, labeled_region):
        ones_tensor = torch.cuda.FloatTensor(entropy_map.shape[0], entropy_map.shape[1]).fill_(0)
        if labeled_region:
            for lr in labeled_region:
                zero_out_mask = ones_tensor != 0
                r0 = lr[0]
                c0 = lr[1]
                r1 = lr[0] + lr[2]
                c1 = lr[1] + lr[3]
                zero_out_mask[r0:r1, c0:c1] = 1
                entropy_map[zero_out_mask] = 0

    def create_region_maps(self, model, images, existing_regions, region_size, selection_size):

        def turn_on_dropout(m):
            if type(m) == nn.Dropout2d:
                m.train()
        model.apply(turn_on_dropout)

        score_maps = torch.cuda.FloatTensor(len(images), self.crop_size - region_size + 1, self.crop_size - region_size + 1)
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size), batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        weights = torch.cuda.FloatTensor(region_size, region_size).fill_(1.)

        map_ctr = 0
        # commented lines are for visualization and verification
        entropy_maps = []
        base_images = []
        for image_batch in tqdm(loader):
            image_batch = image_batch.cuda()
            for img_idx, entropy_map in enumerate(self._get_vote_entropy_for_batch(model, image_batch)):
                ActiveSelectionMCDropout.suppress_labeled_entropy(entropy_map, existing_regions[map_ctr])
                base_images.append(image_batch[img_idx, :, :, :].cpu().numpy())
                entropy_maps.append(entropy_map.cpu().numpy())
                score_maps[map_ctr, :, :] = torch.nn.functional.conv2d(entropy_map.unsqueeze(
                    0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
                map_ctr += 1

        min_val = score_maps.min()
        max_val = score_maps.max()
        minmax_norm = lambda x: (x - min_val) / (max_val - min_val)
        score_maps = minmax_norm(score_maps)

        num_requested_indices = (selection_size * self.crop_size * self.crop_size) / (region_size * region_size)
        regions, num_selected_indices = ActiveSelectionMCDropout.square_nms(score_maps, region_size, num_requested_indices)
        # print(f'Requested/Selected indices {num_requested_indices}/{num_selected_indices}')

        for i in range(len(regions)):
            ActiveSelectionMCDropout._visualize_regions(base_images[i], entropy_maps[i], regions[i], region_size)

        new_regions = {}
        for i in range(len(regions)):
            if regions[i] != []:
                new_regions[images[i]] = regions[i]

        return new_regions, num_selected_indices

    def get_vote_entropy_for_images(self, model, images):

        def turn_on_dropout(m):
            if type(m) == nn.Dropout2d:
                m.train()
        model.apply(turn_on_dropout)

        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size), batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)

        entropies = []
        for image_batch in tqdm(loader):
            image_batch = image_batch.cuda()
            entropies.extend([torch.sum(x).cpu().item() / (image_batch.shape[2] * image_batch.shape[3])
                              for x in self._get_vote_entropy_for_batch(model, image_batch)])

        model.eval()

        return entropies

    @staticmethod
    def _visualize_entropy(image_normalized, entropy_map, prediction):
        import matplotlib
        import matplotlib.pyplot as plt
        image_unnormalized = ((np.transpose(image_normalized, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
        prediction_mapped = map_segmentation_to_colors(prediction.astype(np.uint8), 'cityscapes')
        norm = matplotlib.colors.Normalize(vmin=np.min(entropy_map), vmax=np.max(entropy_map), clip=False)
        plt.figure()
        plt.title('display')
        plt.subplot(1, 3, 1)
        plt.imshow(image_unnormalized)
        plt.subplot(1, 3, 2)
        plt.imshow(prediction_mapped)
        plt.subplot(1, 3, 3)
        plt.imshow(entropy_map, norm=norm, cmap='gray')
        plt.show(block=True)

    @staticmethod
    def _visualize_regions(base_image, image, regions, region_size):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        _, ax = plt.subplots(2)
        if len(base_image.shape) == 3:
            ax[0].imshow(((np.transpose(base_image, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8))
        else:
            ax[0].imshow(base_image)
        ax[1].imshow(image)
        for r in regions:
            rect = patches.Rectangle((r[1], r[0]), region_size, region_size, linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def test_entropy_map_for_images():

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

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 12
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapes(path=dataset_path, base_size=args.base_size,
                                                   crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    val_set = active_cityscapes.ActiveCityscapes(path=dataset_path, base_size=args.base_size,
                                                 crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes',
                                         'al_4-variance-scratch_ep100-bs_125-deeplab-mobilenet-bs_12-513x513', 'run_0425', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_images(model, train_set.current_image_paths[:36]))


def test_nms():
    from PIL import Image
    img_0 = np.asarray(Image.open("resources/images/nms_0.png"), dtype=np.float32) / 256
    img_1 = np.asarray(Image.open("resources/images/nms_1.png"), dtype=np.float32) / 256
    images = [img_0, img_1]

    region_size = 127
    weights = torch.cuda.FloatTensor(region_size, region_size).fill_(1.)
    score_maps = torch.stack([torch.nn.functional.conv2d(torch.from_numpy(img_0).cuda().unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze(
    ).squeeze(), torch.nn.functional.conv2d(torch.from_numpy(img_1).cuda().unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()])
    min_val = score_maps.min()
    max_val = score_maps.max()
    minmax_norm = lambda x: (x - min_val) / (max_val - min_val)
    regions, _ = ActiveSelectionMCDropout._square_nms(minmax_norm(score_maps), region_size, (512 * 512) // (region_size * region_size))
    ActiveSelectionMCDropout._visualize_regions(images[0], images[0], regions[0], region_size)
    ActiveSelectionMCDropout._visualize_regions(images[0], images[1], regions[1], region_size)
    print(regions)


def test_nms_on_entropy_maps():
    from dataloaders.dataset import active_cityscapes
    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_0.txt',
        'batch_size': 12
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapes(path=dataset_path, base_size=args.base_size,
                                                   crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    val_set = active_cityscapes.ActiveCityscapes(path=dataset_path, base_size=args.base_size,
                                                 crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes',
                                         'al_4-variance-scratch_ep100-bs_125-deeplab-mobilenet-bs_12-513x513', 'run_0425', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.create_region_maps(model, train_set.current_image_paths[:12], 127, 4)[0])


def test_create_region_maps_with_region_cityscapes():
    import matplotlib.pyplot as plt
    from dataloaders.dataset import region_cityscapes
    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_dummy.txt',
        'batch_size': 12
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')

    train_set = region_cityscapes.RegionCityscapes(path=dataset_path, base_size=args.base_size,
                                                   crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes',
                                         'al_4-variance-scratch_ep100-bs_125-deeplab-mobilenet-bs_12-513x513', 'run_0425', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, Data Loader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    train_set.image_paths = train_set.image_paths[:3]
    new_regions = active_selector.create_region_maps(model, train_set.image_paths, train_set.get_existing_region_maps(), 127, 1)[0]
    train_set.add_regions(new_regions)
    new_regions = active_selector.create_region_maps(model, train_set.image_paths, train_set.get_existing_region_maps(), 127, 1)[0]
    train_set.add_regions(new_regions)

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    for i, sample in enumerate(dataloader):
        for j in range(sample['image'].size()[0]):
            image = sample['image'].numpy()
            gt = sample['label'].numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('plot')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(gt_colored)

    plt.show(block=True)

if __name__ == '__main__':
    # test_entropy_map_for_images()
    test_create_region_maps_with_region_cityscapes()
