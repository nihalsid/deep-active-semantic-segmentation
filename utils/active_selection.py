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
import math
from sklearn.metrics import pairwise_distances


def get_active_selection_class(active_selection_method, dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
    if active_selection_method == 'coreset':
        return ActiveSelectionCoreSet(dataset_lmdb_env, crop_size, dataloader_batch_size)
    else:
        return ActiveSelectionMCDropout(dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size)


class ActiveSelectionBase:

    def __init__(self, dataset_lmdb_env, crop_size, dataloader_batch_size):
        self.crop_size = crop_size
        self.dataloader_batch_size = dataloader_batch_size
        self.env = dataset_lmdb_env


class ActiveSelectionCoreSet(ActiveSelectionBase):

    def __init__(self,  dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionCoreSet, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)

    def _select_batch(self, features, selected_indices, N):
        new_batch = []
        min_distances = self._updated_distances(selected_indices, features, None)

        for _ in range(N):
            ind = np.argmax(min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in selected_indices
            min_distances = self._updated_distances([ind], features, min_distances)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.5f'
              % max(min_distances))
        return new_batch

    def _updated_distances(self, cluster_centers, features, min_distances):
        x = features[cluster_centers, :]
        dist = pairwise_distances(features, x, metric='euclidean')
        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def get_k_center_greedy_selections(self, selection_size, model, candidate_image_batch, already_selected_image_batch):
        combined_paths = already_selected_image_batch + candidate_image_batch
        loader = DataLoader(paths_dataset.PathsDataset(self.env, combined_paths, self.crop_size),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        FEATURE_DIM = 2736
        features = np.zeros((len(combined_paths), FEATURE_DIM))
        model.eval()
        model.module.set_return_features(True)

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(loader)):
                _, features_batch = model(sample.cuda())
                for feature_idx in range(features_batch.shape[0]):
                    features[batch_idx * self.dataloader_batch_size + feature_idx, :] = features_batch[feature_idx, :, :, :].cpu().numpy().flatten()

        model.module.set_return_features(False)
        selected_indices = self._select_batch(features, list(range(len(already_selected_image_batch))), selection_size)
        return [combined_paths[i] for i in selected_indices]


class ActiveSelectionMCDropout(ActiveSelectionBase):

    def __init__(self, dataset_num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionMCDropout, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.dataset_num_classes = dataset_num_classes

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
        import time
        ones_tensor = torch.FloatTensor(score_maps.shape[1], score_maps.shape[2]).fill_(0)
        selected_indices = [[] for x in range(score_maps.shape[0])]

        tbar = tqdm(list(range(math.ceil(max_selection_count))), desc='NMS')

        selection_count = 0
        for iter_idx in tbar:
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

            if score_maps.max() < 0.1:
                break

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
        minmax_norm = lambda x: x.add_(min_val).mul_(1.0 / (max_val - min_val))
        minmax_norm(score_maps)

        num_requested_indices = (selection_size * self.crop_size * self.crop_size) / (region_size * region_size)
        regions, num_selected_indices = ActiveSelectionMCDropout.square_nms(score_maps.cpu(), region_size, num_requested_indices)
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
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
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
    score_maps = torch.cuda.FloatTensor(500, 386, 386)
    score_maps[:2, :, :] = torch.stack([torch.nn.functional.conv2d(torch.from_numpy(img_0).cuda().unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze(
    ).squeeze(), torch.nn.functional.conv2d(torch.from_numpy(img_1).cuda().unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()])
    min_val = score_maps.min()
    max_val = score_maps.max()
    minmax_norm = lambda x: x.add_(min_val).mul_(1.0 / (max_val - min_val))
    minmax_norm(score_maps)
    regions, _ = ActiveSelectionMCDropout.square_nms(score_maps.cpu(), region_size, (512 * 512) // (region_size * region_size))
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
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
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

    train_set = region_cityscapes.ActiveCityscapesRegion(path=dataset_path, base_size=args.base_size,
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

    region_size = 127
    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    train_set.image_paths = train_set.image_paths[:3]
    print(train_set.get_fraction_of_labeled_data())
    new_regions, counts = active_selector.create_region_maps(model, train_set.image_paths, train_set.get_existing_region_maps(), region_size, 1)
    train_set.expand_training_set(new_regions, counts * region_size * region_size)
    print(train_set.get_fraction_of_labeled_data())
    # new_regions, counts = active_selector.create_region_maps(model, train_set.image_paths, train_set.get_existing_region_maps(), region_size, 1)
    # train_set.expand_training_set(new_regions, counts * region_size * region_size)
    # print(train_set.get_fraction_of_labeled_data())

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


def test_visualize_feature_space():

    from dataloaders.dataset import active_cityscapes
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import json

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_0.txt',
        'batch_size': 12,
        'cuda': True
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, return_features=True)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_0-deeplab-mobilenet-bs12-513x513', 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()
    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

    with open("datasets/cityscapes/clusters/clusters_0.txt", "r") as fptr:
        cluster_dict = json.loads(fptr.read())

    all_paths = []
    cluster_index_length = []
    labels = []
    current_idx = 0
    for cluster in cluster_dict:
        all_paths.extend([u'{}'.format(x.strip()).encode('ascii') for x in cluster_dict[cluster]])
        cluster_index_length.append((current_idx, current_idx + len(cluster_dict[cluster])))
        current_idx = current_idx + len(cluster_dict[cluster])
        labels.append(cluster)

    dataloader = DataLoader(paths_dataset.PathsDataset(train_set.env, all_paths, train_set.crop_size),
                            batch_size=1, shuffle=False, num_workers=0)

    aggregated_features = []

    for sample in tqdm(dataloader):
        with torch.no_grad():
            output, features = model(sample.cuda())
            features = features[0, :, :, :].cpu().numpy()
            aggregated_features.append(features.flatten())

    print(features.shape)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500000)
    tsne_features = tsne.fit_transform(np.array(aggregated_features))
    print(np.array(aggregated_features).shape, tsne_features.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for i in range(len(cluster_index_length)):
        print(cluster_index_length[i][0], cluster_index_length[i][1])
        ax.scatter(tsne_features[cluster_index_length[i][0]:cluster_index_length[i][1], 0], tsne_features[
                   cluster_index_length[i][0]:cluster_index_length[i][1], 1], alpha=0.8, c=colors[i], edgecolors='none', label=labels[i])
    plt.axis('equal')
    plt.tight_layout()
    plt.legend(loc=2, prop={'size': 4})
    plt.show()


def test_core_set():

    from dataloaders.dataset import active_cityscapes
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import json

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_0.txt',
        'batch_size': 12,
        'cuda': True
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_0-deeplab-mobilenet-bs12-513x513', 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()
    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

    with open("datasets/cityscapes/clusters/clusters_0.txt", "r") as fptr:
        cluster_dict = json.loads(fptr.read())

    all_paths = []
    cluster_index_length = []
    labels = []
    current_idx = 0
    for cluster in cluster_dict:
        all_paths.extend([u'{}'.format(x.strip()).encode('ascii') for x in cluster_dict[cluster]])
        cluster_index_length.append((current_idx, current_idx + len(cluster_dict[cluster])))
        current_idx = current_idx + len(cluster_dict[cluster])
        labels.append(cluster)

    active_selection = ActiveSelectionCoreSet(train_set.env, args.crop_size, args.batch_size)
    selected_clusters = active_selection.get_k_center_greedy_selections(10, model, all_paths[1:], [all_paths[0]])


def test_kcenter():
    active_selection = ActiveSelectionCoreSet(None, None, None)
    features = np.array([[1, 1], [2, 2], [2, 4], [3, 3], [4, 2], [4, 5], [5, 4], [6, 2], [7, 6]])
    print(features.shape)
    selected_indices = active_selection._select_batch(features, [6], 5)
    print(selected_indices)

if __name__ == '__main__':
    # test_entropy_map_for_images()
    # test_nms()
    # test_create_region_maps_with_region_cityscapes()
    # test_visualize_feature_space()
    # test_core_set()
    test_kcenter()
