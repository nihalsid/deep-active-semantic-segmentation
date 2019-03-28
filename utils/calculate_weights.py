import os
from tqdm import tqdm
import numpy as np
from constants import DATASET_ROOT


def calculate_weights_labels(dataset, dataloader, num_classes):

	z = np.zeros((num_classes,))

	print ("Calculating class weights..")
	for sample in tqdm(dataloader):
		y = sample['label']
		y = y.detach().cpu().numpy()
		mask = np.logical_and((y >= 0), (y < num_classes))
		labels = y[mask].astype(np.uint8)
		count_l = np.bincount(labels, minlength=num_classes)
		z += count_l

	total_frequency = np.sum(z)
	clas_weights = []

	for frequency in z:
		class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
		clas_weights.append(class_weight)

	ret = np.array(clas_weights)
	class_weights_path = os.path.join(DATASET_ROOT, dataset, 'class_weights.npy')
	np.save(class_weights_path, ret)

	return ret