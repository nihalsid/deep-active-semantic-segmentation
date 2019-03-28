from dataloaders.dataset import cityscapes
from torch.utils.data import DataLoader
from constants import DATASET_ROOT
import os

def make_dataloader(dataset, base_size, crop_size, batch_size, **kwargs):

	if dataset == 'cityscapes':
		dataset_path = os.path.join(DATASET_ROOT, dataset)
		train_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size, split='train')
		val_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size, split='val')
		test_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size, split='test')
		num_classes = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

		return train_loader, val_loader, test_loader, num_classes

	else:
		raise NotImplementedError