import torch
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.misc import imresize
from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))

        mask = mask.astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        return {'image': img,
                'label': mask}


class RandomRotate(object):

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = gaussian_filter(img, sigma=random.random())

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):

    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.shape[1], img.shape[0]

        if w > h:  # h = 1024, w = 2048, oh = 512, ow = 1024
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = imresize(img, (oh, ow))
        mask = imresize(mask, (oh, ow), 'nearest')

        # center crop
        w, h = img.shape[1], img.shape[0]
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img[y1: y1 + self.crop_size, x1: x1 + self.crop_size, :]
        mask = mask[y1: y1 + self.crop_size, x1: x1 + self.crop_size]

        return {'image': img,
                'label': mask}


class Scale(object):

    def __init__(self, base_size):
        self.base_size = base_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.shape[1], img.shape[0]

        if w > h:  # h = 1024, w = 2048, oh = 512, ow = 1024
            oh = self.base_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.base_size
            oh = int(1.0 * h * ow / w)

        img = imresize(img, (oh, ow))
        mask = imresize(mask, (oh, ow), 'nearest')

        return {'image': img,
                'label': mask}


class ScaleImageOnly(object):

    def __init__(self, base_size):
        self.base_size = base_size

    def __call__(self, sample):
        img = sample
        w, h = img.shape[1], img.shape[0]

        if w > h:  # h = 1024, w = 2048, oh = 512, ow = 1024
            oh = self.base_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.base_size
            oh = int(1.0 * h * ow / w)

        img = imresize(img, (oh, ow))

        return img


class ScaleWithPadding(object):

    def __init__(self, base_size):
        self.base_size = base_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        w, h = img.shape[1], img.shape[0]
        out_image = np.zeros((self.base_size, self.base_size, 3), dtype=np.float32)
        out_mask = np.ones((self.base_size, self.base_size), dtype=np.uint8) * 255

        if w < h:  # h = 1024, w = 2048, oh = 512, ow = 1024
            oh = self.base_size
            ow = int(1.0 * w * oh / h)
            if ow % 2 != 0:
                ow += 1
        else:
            ow = self.base_size
            oh = int(1.0 * h * ow / w)
            if oh % 2 != 0:
                oh += 1

        img = imresize(img, (oh, ow))
        mask = imresize(mask, (oh, ow), 'nearest')

        out_image[self.base_size // 2 - oh // 2: self.base_size // 2 + oh // 2, self.base_size // 2 - ow // 2: self.base_size // 2 + ow // 2, :] = img
        out_mask[self.base_size // 2 - oh // 2: self.base_size // 2 + oh // 2, self.base_size // 2 - ow // 2: self.base_size // 2 + ow // 2] = mask

        return {'image': out_image,
                'label': out_mask}


class ScaleWithPaddingImageOnly(object):

    def __init__(self, base_size):
        self.base_size = base_size

    def __call__(self, sample):
        img = sample

        w, h = img.shape[1], img.shape[0]
        out_image = np.zeros((self.base_size, self.base_size, 3), dtype=np.float32)

        if w < h:  # h = 1024, w = 2048, oh = 512, ow = 1024
            oh = self.base_size
            ow = int(1.0 * w * oh / h)
            if ow % 2 != 0:
                ow += 1
        else:
            ow = self.base_size
            oh = int(1.0 * h * ow / w)
            if oh % 2 != 0:
                oh += 1

        img = imresize(img, (oh, ow))

        out_image[self.base_size // 2 - oh // 2: self.base_size // 2 + oh // 2, self.base_size // 2 - ow // 2: self.base_size // 2 + ow // 2, :] = img

        return out_image


class FixScaleCropImageOnly(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        w, h = sample.shape[1], sample.shape[0]
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        sample = imresize(sample, (oh, ow))
        # center crop
        w, h = sample.shape[1], sample.shape[0]
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        sample = sample[y1: y1 + self.crop_size, x1: x1 + self.crop_size, :]
        return sample


def invert_fix_scale_crop(label, output, region, crop_size):

    h, w = label.shape

    if w > h:
        oh = crop_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = crop_size
        oh = int(1.0 * h * ow / w)

    x1 = int(round((ow - crop_size) / 2.))
    y1 = int(round((oh - crop_size) / 2.))

    b0, b1, b2, b3 = round((region[0] + y1) * (h / oh)), round((region[1] + x1) * (w / ow)), round(region[2] * (h / oh)), round(region[3] * (w / ow))
    output[b0: b0 + b2, b1: b1 + b3] = label[b0: b0 + b2, b1: b1 + b3]


def invert_scale_crop(label, output, region, base_size):

    h, w = label.shape

    if w < h:  # h = 1024, w = 2048, oh = 512, ow = 1024
        oh = base_size
        ow = int(1.0 * w * oh / h)
        if ow % 2 != 0:
            ow += 1
    else:
        ow = base_size
        oh = int(1.0 * h * ow / w)
        if oh % 2 != 0:
            oh += 1

    #print('height/width', h, w)
    #print('outH/outW', oh, ow)
    padding_h = abs(base_size - oh) // 2
    padding_w = abs(base_size - ow) // 2
    #print('Padding h/w', padding_h, padding_w)

    b0, b1, b2, b3 = round((max(region[0] - padding_h, 0)) * (h / oh)), round(max((region[1] - padding_w), 0)
                                                                              * (w / ow)), round(region[2] * (h / oh)), round(region[3] * (w / ow))

    output[b0: b0 + b2, b1: b1 + b3] = label[b0: b0 + b2, b1: b1 + b3]


class FixedResize(object):

    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]

        img = imresize(img, self.size)
        mask = imresize(img, self.size, 'nearest')

        return {'image': img,
                'label': mask}
