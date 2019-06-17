from dataloaders.dataset import paths_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm


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

        print('Maximum distance from cluster centers is %0.5f' % max(min_distances))
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
        if model.module.model_name == 'deeplab':
            FEATURE_DIM = 2736
            average_pool_kernel_size = (64, 64)
        elif model.module.model_name == 'enet':
            FEATURE_DIM = 1152
            average_pool_kernel_size = (32, 32)
        features = np.zeros((len(combined_paths), FEATURE_DIM))
        model.eval()
        model.module.set_return_features(True)

        average_pool_stride = average_pool_kernel_size[0] // 2
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(loader)):
                _, features_batch = model(sample.cuda())
                features_batch = F.avg_pool2d(features_batch, average_pool_kernel_size, average_pool_stride)
                for feature_idx in range(features_batch.shape[0]):
                    features[batch_idx * self.dataloader_batch_size + feature_idx, :] = features_batch[feature_idx, :, :, :].cpu().numpy().flatten()

        model.module.set_return_features(False)
        selected_indices = self._select_batch(features, list(range(len(already_selected_image_batch))), selection_size)
        return [combined_paths[i] for i in selected_indices]
