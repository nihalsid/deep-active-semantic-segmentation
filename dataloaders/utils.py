import numpy as np
import torch


def create_cityscapes_label_colormap():

    return {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        255: [255, 255, 255]
    }


def create_binary_colormap():

    return {
        0: [135, 8, 44],
        1: [12, 96, 0],
        255: [255, 255, 255]
    }


def create_pascal_label_colormap():

    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    dict_colormap = {}
    for i in range(256):
        dict_colormap[i] = colormap[i, :].tolist()

    return dict_colormap


def get_colormap(dataset):

    if dataset == 'cityscapes' or dataset == 'active_cityscapes_image' or dataset == 'active_cityscapes_region':
        return create_cityscapes_label_colormap()
    elif dataset == 'binary':
        return create_binary_colormap()
    elif dataset == 'pascal':
        return create_pascal_label_colormap()

    raise Exception('No colormap for dataset found')


def map_segmentations_to_colors(segmentations, dataset):
    rgb_masks = []
    for segmentation in segmentations:
        rgb_mask = map_segmentation_to_colors(segmentation, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def map_binary_output_mask_to_colors(binary_segmentation):
    rgb_masks = []
    for segmentation in binary_segmentation:
        rgb_mask = map_segmentation_to_colors(segmentation, 'binary')
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def map_segmentation_to_colors(segmentation, dataset):
    colormap = get_colormap(dataset)
    colored_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))

    for label in np.unique(segmentation).tolist():
        colored_segmentation[segmentation == label, :] = colormap[label]

    colored_segmentation /= 255.0
    return colored_segmentation
