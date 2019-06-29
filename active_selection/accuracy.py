from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloaders.dataset import paths_dataset
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm
import os
import time
from active_selection.mc_dropout import ActiveSelectionMCDropout


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
        #times = []
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()
                #a = time.time()
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
                #times.append(time.time() - a)
        #print(np.mean(times), np.std(times))
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

    def suppress_labeled_areas(self, score_map, labeled_region):
        ones_tensor = torch.cuda.FloatTensor(score_map.shape[0], score_map.shape[1]).fill_(0)
        if labeled_region:
            for lr in labeled_region:
                zero_out_mask = ones_tensor != 0
                r0 = lr[0]
                c0 = lr[1]
                r1 = lr[0] + lr[2]
                c1 = lr[1] + lr[3]
                zero_out_mask[r0:r1, c0:c1] = 1
                score_map[zero_out_mask] = 0

    def get_least_accurate_region_maps(self, model, images, existing_regions, region_size, selection_size):
        base_size = 512 if self.crop_size == -1 else self.crop_size
        score_maps = torch.cuda.FloatTensor(len(images), base_size - region_size + 1, base_size - region_size + 1)
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size, include_labels=True),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        weights = torch.cuda.FloatTensor(region_size, region_size).fill_(1.)

        map_ctr = 0
        times = []
        # commented lines are for visualization and verification
        #error_maps = []
        #base_images = []
        softmax = torch.nn.Softmax2d()
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda()
                label_batch = sample['label'].cuda()

                a = time.time()
                deeplab_output, unet_output = model(image_batch)
                prediction = softmax(unet_output)
                for idx in range(prediction.shape[0]):
                    mask = (label_batch[idx, :, :] < 0) | (label_batch[idx, :, :] >= self.num_classes)
                    incorrect = prediction[idx, 0, :, :]
                    incorrect[mask] = 0
                    self.suppress_labeled_areas(incorrect, existing_regions[map_ctr])
                    #base_images.append(image_batch[idx, :, :, :].cpu().numpy())
                    # error_maps.append(incorrect.cpu().numpy())
                    score_maps[map_ctr, :, :] = torch.nn.functional.conv2d(incorrect.unsqueeze(
                        0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
                    map_ctr += 1
                times.append(time.time() - a)
        min_val = score_maps.min()
        max_val = score_maps.max()
        minmax_norm = lambda x: x.add_(-min_val).mul_(1.0 / (max_val - min_val))
        minmax_norm(score_maps)

        num_requested_indices = (selection_size * base_size * base_size) / (region_size * region_size)
        b = time.time()
        regions, num_selected_indices = ActiveSelectionMCDropout.square_nms(score_maps.cpu(), region_size, num_requested_indices)
        print(np.mean(times), np.std(times), time.time() - b)
        # print(f'Requested/Selected indices {num_requested_indices}/{num_selected_indices}')

        # for i in range(len(regions)):
        #    ActiveSelectionMCDropout._visualize_regions(base_images[i], error_maps[i], regions[i], score_maps[i, :, :].cpu().numpy(), region_size)

        new_regions = {}
        for i in range(len(regions)):
            if regions[i] != []:
                new_regions[images[i]] = regions[i]

        model.eval()

        return new_regions, num_selected_indices

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
