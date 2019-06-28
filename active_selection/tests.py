import constants
import numpy as np
import random
from dataloaders.dataset import paths_dataset
from dataloaders.dataset import active_cityscapes
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
from models.accuracy_predictor import *
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
from active_selection.mc_noise import ActiveSelectionMCNoise
from active_selection.accuracy import ActiveSelectionAccuracy


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
                                                        crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
    # val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
    #                                                  crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'alefw_1-mc_vote_entropy_images-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0002', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_images(model, train_set.current_image_paths[:2], 1))


def test_entropy_map_for_images_enet():

    args = {
        'base_size': 512,
        'crop_size': -1,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 1
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                      crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
    from models.enet import ENet
    model = ENet(num_classes=train_set.NUM_CLASSES, encoder_relu=True, decoder_relu=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_efw-enetf-bs_4-no_sched-1024x512-lr_0.01\\experiment_3', 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_images(model, train_set.current_image_paths[:10], 5))


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

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'eval_0-random_images-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0002', 'best.pth.tar'))
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


def test_create_region_maps_with_region_pascal():
    import matplotlib.pyplot as plt
    from dataloaders.dataset import region_pascal
    args = {
        'base_size': 512,
        'crop_size': -1,
        'seed_set': 'set_dummy.txt',
        'batch_size': 5
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'pascal')

    train_set = region_pascal.ActivePascalRegion(path=dataset_path, base_size=args.base_size,
                                                 crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_pascal_region',
                                         'evalpa_5-noise_variance_entropy_regions_ep150-abs_60-deeplab-mobilenet-bs_5-512x512-lr_0.007', 'run_0003', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, Data Loader(train_set, batch_size=2, shuffle=False), args)

    region_size = 129
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
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'pascal')
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
                                         'base_efw-deeplab-mobilenet-bs_5-512x512-lr_0.01', 'model_best.pth.tar'))
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


def test_core_set_enet():

    from dataloaders.dataset import active_cityscapes
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import json

    args = {
        'base_size': 512,
        'crop_size': -1,
        'seed_set': 'set_0.txt',
        'batch_size': 5,
        'cuda': True
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    from models.enet import ENet
    model = ENet(num_classes=train_set.NUM_CLASSES, encoder_relu=True, decoder_relu=True)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_efw-enetf-bs_4-no_sched-1024x512-lr_0.01\\experiment_3', 'checkpoint.pth.tar'))
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
        'seed_set': 'set_0.txt',
        'batch_size': 12,
        'cuda': True
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)

    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'alefw_10-ceal_confidence-increment_scratch_ep200-bs_125-deeplab-mobilenet-bs_5-513x513', 'run_0002', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    active_selector = ActiveSelectionCEAL(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_least_confident_samples(model, train_set.current_image_paths[:2], 1))
    # print(active_selector.get_least_margin_samples(model, train_set.current_image_paths[:10], 5))
    # print(active_selector.get_maximum_entropy_samples(model, train_set.current_image_paths[:10], 5))
    # print(active_selector.get_fusion_of_confidence_margin_entropy_samples(model, train_set.current_image_paths[:20], 3))
    #weak_labels = active_selector.get_weakly_labeled_data(model, train_set.remaining_image_paths[:50], 0.70)
    # train_set.add_weak_labels(weak_labels)

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
            # plt.show(block=True)


def test_max_set_cover():

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    num_images = 10
    np.random.seed(seed=27)
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
        random_candidates = list(np.random.randint(0, len(image_features), big_k))  # random.sample(range(len(image_features)), big_k)
        candidate_image_features = list(images[random_candidates, :])
        selected_indices = ActiveSelectionMaxSubset(None, None, None)._max_representative_samples(image_features, candidate_image_features, small_k)
        selected_indices_across_iters.extend([random_candidates[i] for i in selected_indices])
        image_features_idx = [i for i in range(len(image_features)) if i not in selected_indices_across_iters]
        image_features = list(images[image_features_idx, :])
        print([random_candidates[i] for i in selected_indices])
        for j in range(-1, small_k):
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


def test_image_features_enet():
    args = {
        'base_size': 512,
        'crop_size': -1,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 5
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                      crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)
    from models.enet import ENet
    model = ENet(num_classes=train_set.NUM_CLASSES, encoder_relu=True, decoder_relu=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_efw-enetf-bs_4-no_sched-1024x512-lr_0.01\\experiment_3', 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    active_selector = ActiveSelectionMCDropout(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    max_subset_selector = ActiveSelectionMaxSubset(train_set.env, args.crop_size, args.batch_size)
    candidates = active_selector.get_vote_entropy_for_images(model, train_set.current_image_paths[:36], 8)
    max_subset_selector.get_representative_images(model, train_set.current_image_paths[:36], candidates)


def test_entropy_map_for_images_with_inoise():

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
    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=False)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes',
                                         'al_4-variance-scratch_ep100-bs_125-deeplab-mobilenet-bs_12-513x513', 'run_0425', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCNoise(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_images_with_input_noise(model, train_set.current_image_paths[:36], 10))


def test_entropy_map_for_images_with_fnoise():

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
    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=False)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes',
                                         'al_4-variance-scratch_ep100-bs_125-deeplab-mobilenet-bs_12-513x513', 'run_0425', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCNoise(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_images_with_feature_noise(model, train_set.current_image_paths[:36], 10))


def test_entropy_map_for_images_with_fnoise_enet():

    args = {
        'base_size': 512,
        'crop_size': -1,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 12
    }
    from models.enet import ENet

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    model = ENet(num_classes=train_set.NUM_CLASSES, encoder_relu=True, decoder_relu=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_efw-enetf-bs_4-no_sched-1024x512-lr_0.01\\experiment_3', 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCNoise(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_images_with_feature_noise(model, train_set.current_image_paths[:36], 10))


def test_entropy_map_for_images_with_noise_and_ve():

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

    active_selector = ActiveSelectionMCNoise(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_batch_with_noise_and_vote_entropy(model, train_set.current_image_paths[:36], 10))


def test_entropy_map_for_images_with_noise_and_ve_enet():

    args = {
        'base_size': 512,
        'crop_size': -1,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 12
    }
    from models.enet import ENet

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)
    model = ENet(num_classes=train_set.NUM_CLASSES, encoder_relu=True, decoder_relu=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'cityscapes',
                                         'base_efw-enetf-bs_4-no_sched-1024x512-lr_0.01\\experiment_3', 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, DataLoader(train_set, batch_size=2, shuffle=False), args)

    active_selector = ActiveSelectionMCNoise(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_vote_entropy_for_batch_with_noise_and_vote_entropy(model, train_set.current_image_paths[:36], 10))


def test_accuracy_selector():

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
    model = DeepLab(num_classes=train_set.NUM_CLASSES, backbone='mobilenet', output_stride=16, sync_bn=False, freeze_bn=False, mc_dropout=True)

    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes',
                                         'al_4-variance-scratch_ep100-bs_125-deeplab-mobilenet-bs_12-513x513', 'run_0425', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()

    active_selector = ActiveSelectionAccuracy(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_least_accurate_sample_using_labels(model, train_set.current_image_paths[:36], 10))


def test_accuracy_est_selector():
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors

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

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image', 'accuracy_predictor_50_point_fit', 'run_0002', 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        print(train_set.current_image_paths[i])
        image_batch = sample['image'].cuda()
        dl_out, un_out = model(image_batch)
        un_target = dl_out.argmax(1).cpu().squeeze() == sample['label'].long()
        un_target[sample['label'] == 255] = 255
        for j in range(sample['image'].size()[0]):
            image = sample['image'].cpu().numpy()
            gt = sample['label'].cpu().numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('plot')
            plt.subplot(231)
            plt.imshow(image_unnormalized)
            plt.subplot(232)
            plt.imshow(map_segmentation_to_colors(np.array(un_target.numpy()[j]).astype(np.uint8), 'binary'))
            plt.subplot(233)
            plt.imshow(gt_colored)
            plt.subplot(236)
            plt.imshow(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
            plt.subplot(235)
            plt.imshow(map_segmentation_to_colors(np.array(un_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'binary'))
            # plt.show(block=True)

        if i == 1:
            break

    active_selector = ActiveSelectionAccuracy(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_least_accurate_samples(model, train_set.current_image_paths[:10], 5, 'softmax'))


def test_noisy_create_region_maps_with_region_cityscapes():
    import matplotlib.pyplot as plt
    from dataloaders.dataset import region_cityscapes
    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_dummy.txt',
        'batch_size': 2
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
    active_selector = ActiveSelectionMCNoise(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
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


def test_gradient_visualization():
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': '',
        'seed_set': 'set_dummy.txt',
        'batch_size': 1
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'alefw_7-accuracy_prediction-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0031', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    softmax = nn.Softmax(dim=1)

    for i, sample in enumerate(dataloader):
        print(train_set.current_image_paths[i])
        image_batch = sample['image'].cuda()
        #image_batch.requires_grad = True
        dl_out, un_out = model(image_batch)
        un_in = torch.cuda.FloatTensor(torch.cat([softmax(dl_out), image_batch], dim=1).detach().cpu().numpy())
        un_in.requires_grad = True
        un_clean_out = model.module.unet(un_in)
        un_clean_out[0, 1, :, :].backward(torch.ones_like(un_clean_out[0, 1, :, :]).cuda())

        grad = un_in.grad.sum(1)  # torch.norm(un_in.grad, p=2, dim=1)
        min_val = grad.min()
        max_val = grad.max()
        minmax_norm = lambda x: x.add_(-min_val).mul_(1.0 / (max_val - min_val))
        minmax_norm(grad)
        un_target = dl_out.argmax(1).cpu().squeeze() == sample['label'].long()
        un_target[sample['label'] == 255] = 255
        for j in range(sample['image'].size()[0]):
            image = sample['image'].cpu().numpy()
            gt = sample['label'].cpu().numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('plot')
            plt.subplot(231)
            plt.imshow(image_unnormalized)
            plt.subplot(232)
            plt.imshow(map_segmentation_to_colors(np.array(un_target.numpy()[j]).astype(np.uint8), 'binary'))
            plt.subplot(233)
            plt.imshow(gt_colored)
            plt.subplot(234)
            plt.imshow(grad[j, :, :].cpu().numpy(), cmap='gray')
            plt.subplot(236)
            plt.imshow(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
            plt.subplot(235)
            plt.imshow(map_segmentation_to_colors(np.array(un_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'binary'))
            plt.show(block=True)

        if i == 1:
            break


def test_gradient_selection():
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 1
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'albnf_15-accuracy_prediction-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0031', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    softmax = nn.Softmax(dim=1)

    for i, sample in enumerate(dataloader):
        image_batch = sample['image'].cuda()
        dl_out, un_out = model(image_batch)
        un_target = dl_out.argmax(1).cpu().squeeze() == sample['label'].long()
        un_target[sample['label'] == 255] = 255
        print(train_set.current_image_paths[i], (un_target == 0).sum())
        for j in range(sample['image'].size()[0]):
            image = sample['image'].cpu().numpy()
            gt = sample['label'].cpu().numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('plot')
            plt.subplot(231)
            plt.imshow(image_unnormalized)
            plt.subplot(232)
            plt.imshow(map_segmentation_to_colors(np.array(un_target.numpy()[j]).astype(np.uint8), 'binary'))
            plt.subplot(233)
            plt.imshow(gt_colored)
            plt.subplot(236)
            plt.imshow(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
            plt.subplot(235)
            plt.imshow(map_segmentation_to_colors(np.array(un_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'binary'))
            # plt.show(block=True)

        if i == 10:
            break

    active_selector = ActiveSelectionAccuracy(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_adversarially_vulnarable_samples(model, train_set.current_image_paths[:10], 5))


def test_unsure_samples():
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 1
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'albnf_15-accuracy_prediction-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0031', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            image_batch = sample['image'].cuda()
            dl_out, un_out = model(image_batch)
            prediction = softmax(un_out)
            un_target = dl_out.argmax(1).cpu().squeeze() == sample['label'].long()
            un_target[sample['label'] == 255] = 255
            unsure = 4 * prediction[:, 1, :, :] - 4 * prediction[:, 1, :, :] ** 2
            print(train_set.current_image_paths[i], (un_target == 0).sum())
            for j in range(sample['image'].size()[0]):
                image = sample['image'].cpu().numpy()
                gt = sample['label'].cpu().numpy()
                gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
                image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
                plt.figure()
                plt.title('plot')
                plt.subplot(231)
                plt.imshow(image_unnormalized)
                plt.subplot(232)
                plt.imshow(map_segmentation_to_colors(np.array(un_target.numpy()[j]).astype(np.uint8), 'binary'))
                plt.subplot(233)
                plt.imshow(gt_colored)
                plt.subplot(234)
                min_val = unsure[j, :, :].min()
                max_val = unsure[j, :, :].max()
                minmax_norm = lambda x: x.add_(-min_val).mul_(1.0 / (max_val - min_val))
                minmax_norm(unsure[j, :, :])
                plt.imshow(unsure[j, :, :].cpu().numpy(), cmap='gray')
                plt.subplot(236)
                plt.imshow(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
                plt.subplot(235)
                plt.imshow(map_segmentation_to_colors(np.array(un_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'binary'))
                plt.show(block=True)

            if i == 10:
                break

    active_selector = ActiveSelectionAccuracy(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_unsure_samples(model, train_set.current_image_paths[:10], 5))


def test_create_inaccuracy_maps_with_region_cityscapes():
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

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'alefw_7-accuracy_prediction-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0031', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # ensure that the loaded model is not crap
    # validation(model, Data Loader(train_set, batch_size=2, shuffle=False), args)

    region_size = 127
    active_selector = ActiveSelectionAccuracy(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    train_set.image_paths = train_set.image_paths[:5]
    print(train_set.get_fraction_of_labeled_data())
    new_regions, counts = active_selector.get_least_accurate_region_maps(model, train_set.image_paths, train_set.get_existing_region_maps(), region_size, 0.5)
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
            plt.subplot(121)
            plt.imshow(image_unnormalized)
            plt.subplot(122)
            plt.imshow(gt_colored)

    plt.show(block=True)


def test_inaccuracy_heatmaps():
    import matplotlib
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': '',
        'seed_set': 'set_0.txt',
        'batch_size': 2
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='val', init_set=args.seed_set, overfit=False)

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image',
                                         'alefw_7-accuracy_prediction-scratch_ep200-abs_125-deeplab-mobilenet-bs_5-513x513-lr_0.01', 'run_0031', 'best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    softmax = nn.Softmax(dim=1)

    rgb_images = []
    sem_gt_images = []
    sem_pred_images = []
    acc_gt_images = []
    acc_pred_images = []

    for i, sample in enumerate(dataloader):
        print(train_set.current_image_paths[i])
        image_batch = sample['image'].cuda()
        dl_out, un_out = model(image_batch)
        un_target = dl_out.argmax(1).cpu().squeeze() == sample['label'].long()
        un_target[sample['label'] == 255] = 255
        for j in range(sample['image'].size()[0]):
            image = sample['image'].cpu().numpy()
            gt = sample['label'].cpu().numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('plot')
            plt.subplot(231)
            plt.imshow(image_unnormalized)
            rgb_images.append(image_unnormalized)
            plt.subplot(232)
            np_un_target = np.array(un_target.numpy()[j]).astype(np.float32)
            masked_target_array = np.ma.array(np_un_target, mask=np_un_target == 255)
            masked_target_array = 1 - masked_target_array
            cmap = matplotlib.cm.jet
            cmap.set_bad('white', 1.)
            plt.imshow(masked_target_array, cmap=cmap)
            acc_gt_images.append(cmap(masked_target_array))
            #plt.imshow(map_segmentation_to_colors(np.array(un_target.numpy()[j]).astype(np.uint8), 'binary'))
            plt.subplot(233)
            plt.imshow(gt_colored)
            sem_gt_images.append(gt_colored)
            plt.subplot(236)
            plt.imshow(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
            sem_pred_images.append(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
            plt.subplot(235)
            np_un_out = np.array(softmax(un_out.detach()).cpu().numpy()[j, 0, :, :])
            masked_out_array = np.ma.array(np_un_out, mask=np_un_target == 255)
            plt.imshow(masked_out_array, cmap=cmap)
            acc_pred_images.append(cmap(masked_out_array))
            # plt.show(block=True)

        if i == 5:
            break

    for prefix, arr in zip(['rgb', 'sem_gt', 'sem_pred', 'acc_gt', 'acc_pred'], [rgb_images, sem_gt_images, sem_pred_images, acc_gt_images, acc_pred_images]):
        stacked_image = np.ones(((arr[0].shape[0] + 20) * len(arr), arr[0].shape[1], arr[0].shape[2]),
                                dtype=arr[0].dtype) * (255 if arr[0].dtype == np.uint8 else 1)
        for i, im in enumerate(arr):
            stacked_image[i * (arr[0].shape[0] + 20): i * (arr[0].shape[0] + 20) + arr[0].shape[0], :, :] = im
        plt.imsave('%s.png' % (prefix), stacked_image)

    active_selector = ActiveSelectionAccuracy(train_set.NUM_CLASSES, train_set.env, args.crop_size, args.batch_size)
    print(active_selector.get_least_accurate_samples(model, train_set.current_image_paths[:10], 5, 'softmax'))


if __name__ == '__main__':
    test_entropy_map_for_images()
    # test_nms()
    # test_create_region_maps_with_region_cityscapes()
    # test_visualize_feature_space()
    # test_core_set()
    # test_kcenter()
    test_ceal()
    # test_max_set_cover()
    # test_region_features()
    # test_image_features()
    # test_entropy_map_for_images_with_noise_and_ve()
    # test_accuracy_selector()
    # test_noisy_create_region_maps_with_region_cityscapes()
    # test_accuracy_est_selector()
    # test_gradient_selection()
    # test_gradient_visualization()
    # test_unsure_samples()
    # test_create_inaccuracy_maps_with_region_cityscapes()
    # test_create_region_maps_with_region_pascal()
    # test_image_features_enet()
    # test_entropy_map_for_images_enet()
    # test_inaccuracy_heatmaps()
    # test_create_inaccuracy_maps_with_region_cityscapes()
