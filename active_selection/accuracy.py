from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloaders.dataset import paths_dataset
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm
import os
import time


class ActiveSelectionAccuracy(ActiveSelectionBase):

    def __init__(self, num_classes, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionAccuracy, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)
        self.num_classes = num_classes

    def get_least_accurate_sample_using_labels(self, model, images, selection_count):

        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        num_inaccurate_pixels = []

        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                output = model(image_batch)
                prediction = torch.argmax(output, dim=1).type(torch.cuda.FloatTensor)
                for idx in range(prediction.shape[0]):
                    mask = (label_batch[idx, :, :] >= 0) & (label_batch[idx, :, :] < self.num_classes)
                    incorrect = label_batch[idx, mask] != prediction[idx, mask]
                    num_inaccurate_pixels.append(incorrect.sum().cpu().float().item())

        selected_samples = list(zip(*sorted(zip(num_inaccurate_pixels, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def get_least_accurate_samples(self, model, images, selection_count, mode='softmax'):

        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        num_inaccurate_pixels = []
        softmax = torch.nn.Softmax2d()
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                deeplab_output, unet_output = model(image_batch)
                if mode == 'softmax':
                    prediction = softmax(unet_output)
                    for idx in range(prediction.shape[0]):
                        mask = (label_batch[idx, :, :] >= 0) & (label_batch[idx, :, :] < self.num_classes)
                        incorrect = prediction[idx, 0, mask]
                        num_inaccurate_pixels.append(incorrect.sum().cpu().float().item())
                elif mode == 'argmax':
                    prediction = unet_output.argmax(1).squeeze().type(torch.cuda.FloatTensor)
                    for idx in range(prediction.shape[0]):
                        mask = (label_batch[idx, :, :] >= 0) & (label_batch[idx, :, :] < self.num_classes)
                        incorrect = 1 - prediction[idx, mask]
                        num_inaccurate_pixels.append(incorrect.sum().cpu().float().item())
                else:
                    raise NotImplementedError
        selected_samples = list(zip(*sorted(zip(num_inaccurate_pixels, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def get_adversarially_vulnarable_samples(self, model, images, selection_count):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)

        softmax = torch.nn.Softmax2d()
        scores = []
        for sample in tqdm(loader):
            image_batch = sample['image'].cuda()
            label_batch = sample['label'].cuda()
            with torch.no_grad():
                deeplab_output, unet_output = model(image_batch)
            prediction = softmax(unet_output)
            unet_input = torch.cuda.FloatTensor(torch.cat([softmax(deeplab_output), image_batch], dim=1).detach().cpu().numpy())
            unet_input.requires_grad = True
            only_unet_output = model.module.unet(unet_input)
            only_unet_output.backward(torch.ones_like(only_unet_output).cuda())
            gradient_norms = torch.norm(unet_input.grad, p=2, dim=1)
            for idx in range(prediction.shape[0]):
                mask = (label_batch[idx, :, :] < 0) | (label_batch[idx, :, :] >= self.num_classes)
                gradient_norms[idx, mask] = 0
                scores.append(gradient_norms[idx, :, :].mean().cpu().float().item())
        selected_samples = list(zip(*sorted(zip(scores, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        return selected_samples

    def get_unsure_samples(self, model, images, selection_count):
        model.eval()
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)

        softmax = torch.nn.Softmax2d()
        scores = []
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                deeplab_output, unet_output = model(image_batch)
                prediction = softmax(unet_output)
                for idx in range(prediction.shape[0]):
                    mask = (label_batch[idx, :, :] >= 0) & (label_batch[idx, :, :] < self.num_classes)
                    y = 4 * prediction[idx, 1, mask] - 4 * prediction[idx, 1, mask] ** 2
                    scores.append(y.mean().cpu().float().item())
        selected_samples = list(zip(*sorted(zip(scores, images), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        print(scores)
        return selected_samples

    def wait_for_selected_samples(self, location_to_monitor, images):

        while True:
            if os.path.exists(location_to_monitor):
                break
            time.sleep(5)

        paths = []
        with open(location_to_monitor, "r") as fptr:
            paths = [u'{}'.format(x.strip()).encode('ascii') for x in fptr.readlines() if x is not '']

        selected_samples = [x for x in paths if x in images]
        return selected_samples
