import os
from tqdm import tqdm
import numpy as np
from constants import DATASET_ROOT


def calculate_weights_labels(dataset, dataloader, num_classes):

    z = np.zeros((num_classes,))

    print("Calculating class weights..")
    for sample in tqdm(dataloader):
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = np.logical_and((y >= 0), (y < num_classes))
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    z = np.log(z)
    total_frequency = np.sum(z)
    class_weights = []

    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    print('Class weights: ')
    print(class_weights)
    ret = np.array(class_weights)
    return ret
