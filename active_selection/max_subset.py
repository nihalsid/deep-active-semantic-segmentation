from dataloaders.dataset import paths_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from active_selection.base import ActiveSelectionBase
import math
from tqdm import tqdm


class ActiveSelectionMaxSubset(ActiveSelectionBase):

    def __init__(self, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionMaxSubset, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)

    def _max_representative_samples(self, image_features, candidate_image_features, selection_count):
        all_distances = pairwise_distances(image_features, candidate_image_features, metric='euclidean')
        selected_sample_indices = []
        print('Finding max representative candidates..')
        minimum_distances = np.ones((len(image_features))) * float('inf')
        for _ in tqdm(range(selection_count)):
            current_best_score = float("-inf")
            current_best_idx = None
            current_minimum_distances = None
            for i in range(len(candidate_image_features)):
                if i not in selected_sample_indices:
                    selected_sample_indices.append(i)
                    # tmp_distances = np.min(all_distances[:, selected_sample_indices], axis=1)  # np.minimum(current_minimum_distances, all_distances[:, i])
                    tmp_distances = np.minimum(minimum_distances, all_distances[:, i])
                    tmp_score = np.sum(tmp_distances) * -1
                    if tmp_score > current_best_score:
                        current_best_score = tmp_score
                        current_minimum_distances = tmp_distances
                        current_best_idx = i
                    selected_sample_indices.pop()
            selected_sample_indices.append(current_best_idx)
            minimum_distances = current_minimum_distances
        return selected_sample_indices

    def _convert_regions_to_list(self, regions):
        list_images, list_regions = [], []
        for ir in sorted(list(regions.keys())):
            for r in regions[ir]:
                list_images.append(ir)
                list_regions.append(r)
        return list_images, list_regions

    def _get_features_for_image_regions(self, model, images, region_size):
        features = []
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()
        model.module.set_return_features(True)
        with torch.no_grad():
            for batch_idx, image_batch in enumerate(tqdm(loader)):
                image_batch = image_batch.cuda()
                _, features_batch = model(image_batch)
                h = math.floor(region_size * features_batch.shape[2] / self.crop_size)
                w = math.floor(region_size * features_batch.shape[3] / self.crop_size)
                num_rows = math.floor(features_batch.shape[2] / h)
                num_cols = math.floor(features_batch.shape[3] / w)
                for feature_idx in range(features_batch.shape[0]):
                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):
                            row_start = row_idx * h
                            col_start = col_idx * w
                            features.append(F.avg_pool2d(features_batch[feature_idx, :, row_start: row_start + h,
                                                                        col_start: col_start + w], (features_batch.shape[2], features_batch.shape[3])).squeeze().cpu().numpy())
        model.module.set_return_features(False)
        return features

    def _get_features_for_images(self, model, images):
        features = []
        loader = DataLoader(paths_dataset.PathsDataset(self.env, images, self.crop_size),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()
        model.module.set_return_features(True)
        average_pool_kernel_size = (64, 64)
        average_pool_stride = average_pool_kernel_size[0] // 2
        with torch.no_grad():
            for batch_idx, image_batch in enumerate(tqdm(loader)):
                image_batch = image_batch.cuda()
                _, features_batch = model(image_batch)
                for feature_idx in range(features_batch.shape[0]):
                    features.append(F.avg_pool2d(features_batch[feature_idx, :, :, :], average_pool_kernel_size,
                                                 average_pool_stride).squeeze().cpu().numpy().flatten())
        model.module.set_return_features(False)
        return features

    def _get_features_for_regions(self, model, list_images, list_regions):
        features = []
        loader = DataLoader(paths_dataset.PathsDataset(self.env, list_images, self.crop_size),
                            batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
        model.eval()
        model.module.set_return_features(True)
        with torch.no_grad():
            for batch_idx, image_batch in enumerate(tqdm(loader)):
                image_batch = image_batch.cuda()
                _, features_batch = model(image_batch)
                resize_ratio_r = features_batch.shape[2] / self.crop_size
                resize_ratio_c = features_batch.shape[3] / self.crop_size
                for feature_idx in range(features_batch.shape[0]):
                    region = list_regions[batch_idx * self.dataloader_batch_size + feature_idx]
                    r = math.floor(region[0] * resize_ratio_r)
                    c = math.floor(region[1] * resize_ratio_c)
                    h = math.floor(region[2] * resize_ratio_r)
                    w = math.floor(region[3] * resize_ratio_c)
                    features.append(F.avg_pool2d(features_batch[feature_idx, :, r: r + h, c: c + w],
                                                 (features_batch.shape[2], features_batch.shape[3])).squeeze().cpu().numpy())

        model.module.set_return_features(False)
        return features

    def get_representative_regions(self, model, all_images, candidate_regions, region_size):
        candidate_list_images, candidate_list_regions = self._convert_regions_to_list(candidate_regions)
        print('Getting features for images for representativeness ..')
        all_image_features = self._get_features_for_image_regions(model, all_images, region_size)
        print('Getting features for candidates for representativeness ..')
        region_features = self._get_features_for_regions(model, candidate_list_images, candidate_list_regions)
        selected_candidate_indices = self._max_representative_samples(all_image_features, region_features, len(region_features) // 2)
        # self._visualize_selections(all_image_features, region_features, [region_features[i] for i in selected_candidate_indices])
        selected_regions = {}
        for i in selected_candidate_indices:
            if not candidate_list_images[i] in selected_regions:
                selected_regions[candidate_list_images[i]] = []
            selected_regions[candidate_list_images[i]].append(candidate_list_regions[i])
        return selected_regions, len(selected_candidate_indices)

    def get_representative_images(self, model, all_images, candidate_images):
        print('Getting features for images for representativeness ..')
        all_image_features = self._get_features_for_images(model, all_images)
        candidate_features = self._get_features_for_images(model, candidate_images)
        selected_candidate_indices = self._max_representative_samples(all_image_features, candidate_features, len(candidate_features) // 2)
        # self._visualize_selections(all_image_features, candidate_features, [candidate_features[i] for i in selected_candidate_indices])
        return [candidate_images[i] for i in selected_candidate_indices]

    def _visualize_selections(self, all_features, candidate_features, selected_features):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        images = np.concatenate((np.array(all_features), np.array(candidate_features), np.array(selected_features)), axis=0)
        pca = PCA(n_components=2)
        pca_img = pca.fit_transform(images)
        plt.figure('Most representative')
        plt.scatter(pca_img[0:len(all_features), 0], pca_img[0:len(all_features), 1], c='b')
        plt.scatter(pca_img[len(all_features):len(all_features) + len(candidate_features), 0],
                    pca_img[len(all_features):len(all_features) + len(candidate_features), 1], c='g')
        plt.scatter(pca_img[len(all_features) + len(candidate_features):, 0], pca_img[len(all_features) + len(candidate_features):, 1], c='r')
        plt.show()
