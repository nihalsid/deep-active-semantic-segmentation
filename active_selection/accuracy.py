from torch.utils.data import DataLoader
import torch
import numpy as np
from active_selection.base import ActiveSelectionBase
from tqdm import tqdm


class ActiveSelectionAccuracy(ActiveSelectionBase):

	def __init__(self, dataset_lmdb_env, crop_size, dataloader_batch_size):
        super(ActiveSelectionAccuracy, self).__init__(dataset_lmdb_env, crop_size, dataloader_batch_size)

    def get_least_accurate_sample_using_labels(self, model, images, selection_count):
    	
    	model.eval()
    	loader = DataLoader(active_cityscapes.PathsDataset(self.env, images, self.crop_size, include_labels=True), batch_size=self.dataloader_batch_size, shuffle=False, num_workers=0)
    	accuracy = []

    	with torch.no_grad():
    		for image_batch in tqdm(loader):
    			image_batch = sample['image'].cuda()
    			label_batch = sample['label'].cuda()
    			output = model(image_batch)
    			prediction = np.argmax(output, axis=1)
    			correct = label_batch == prediction
    			for idx in range(prediction.shape[0]):
    				accuracy.append(correct[idx, :, :].sum().cpu().item() / (correct.shape[1] * correct.shape[2]))

    	selected_samples = list(zip(*sorted(zip(accuracy, images), key=lambda x: x[0], reverse=False)))[1][:selection_count]
        return selected_samples


