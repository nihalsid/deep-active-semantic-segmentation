import constants
import numpy as np
import random
from dataloaders.dataset import paths_dataset
from dataloaders.dataset import active_cityscapes
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from dataloaders.utils import map_segmentation_to_colors
import math
from active_selection.ceal import ActiveSelectionCEAL
from active_selection.core_set import ActiveSelectionCoreSet
from active_selection.mc_dropout import ActiveSelectionMCDropout
from active_selection.max_subset import ActiveSelectionMaxSubset


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


def test_ceal():
    from dataloaders.dataset import active_cityscapes
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import json

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_dummy.txt',
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
    active_selector = ActiveSelectionCEAL(train_set.env, args.crop_size, args.batch_size)
    # print(active_selector.get_least_confident_samples(model, train_set.current_image_paths[:20], 3))
    # print(active_selector.get_least_margin_samples(model, train_set.current_image_paths[:20], 3))
    # print(active_selector.get_maximum_entropy_samples(model, train_set.current_image_paths[:20], 3))
    # print(active_selector.get_fusion_of_confidence_margin_entropy_samples(model, train_set.current_image_paths[:20], 3))
    weak_labels = active_selector.get_weakly_labeled_data(model, train_set.remaining_image_paths[:50], 0.70)
    train_set.add_weak_labels(weak_labels)

    dataloader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=0)
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


def test_max_set_cover():

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    num_images = 10

    images_cluster1 = np.random.normal(loc=2.0, scale=1.0, size=(400, 1024))
    images_cluster2 = np.random.normal(loc=4.0, scale=1.0, size=(400, 1024))
    images_cluster3 = np.random.normal(loc=6.0, scale=1.0, size=(150, 1024))
    images_outliers = np.random.normal(loc=4.0, scale=3.0, size=(50, 1024))

    images = np.concatenate((images_cluster1, images_cluster2, images_cluster3, images_outliers), axis=0)

    pca = PCA(n_components=2)

    pca_img = pca.fit_transform(images)

    big_k = 8
    small_k = 4
    num_iterations = 1

    image_features = list(images)
    selected_indices_across_iters = []

    for i in range(num_iterations):
        random_candidates = random.sample(range(len(image_features)), big_k)
        candidate_image_features = list(images[random_candidates, :])
        selected_indices = ActiveSelectionMaxSubset(None, None, None)._max_representative_samples(image_features, candidate_image_features, small_k)
        selected_indices_across_iters.extend([random_candidates[i] for i in selected_indices])
        image_features_idx = [i for i in range(len(image_features)) if i not in selected_indices_across_iters]
        image_features = list(images[image_features_idx, :])
        print([random_candidates[i] for i in selected_indices])
        for j in range(small_k):
            plt.figure(i)
            plt.scatter(pca_img[image_features_idx, 0], pca_img[image_features_idx, 1], c='b')
            plt.scatter(pca_img[random_candidates, 0], pca_img[random_candidates, 1], c='g')
            plt.scatter(pca_img[selected_indices_across_iters[:j + 1], 0], pca_img[selected_indices_across_iters[:j + 1], 1], c='r')
            plt.show()


def test_region_features():
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

    region_size = 129
    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    train_set.image_paths = train_set.image_paths[:3]
    new_regions, counts = active_selector.create_region_maps(model, train_set.image_paths, train_set.get_existing_region_maps(), region_size, 2)
    max_subset_selector = ActiveSelectionMaxSubset(train_set.env, args.crop_size, args.batch_size)
    train_set.expand_training_set(max_subset_selector.get_representative_regions(
        model, train_set.image_paths, new_regions, region_size)[0], counts * region_size * region_size)
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


def test_image_features():
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

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    max_subset_selector = ActiveSelectionMaxSubset(train_set.env, args.crop_size, args.batch_size)
    candidates = active_selector.get_vote_entropy_for_images(model, train_set.current_image_paths[:36], 8)
    max_subset_selector.get_representative_images(model, train_set.current_image_paths[:36], candidates)

if __name__ == '__main__':
    # test_entropy_map_for_images()
    # test_nms()
    # test_create_region_maps_with_region_cityscapes()
    # test_visualize_feature_space()
    # test_core_set()
    # test_kcenter()
    # test_ceal()
    # test_max_set_cover()
    # test_region_features()
    test_image_features()
