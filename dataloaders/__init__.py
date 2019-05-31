from dataloaders.dataset import cityscapes, active_cityscapes, region_cityscapes
from torch.utils.data import DataLoader
from constants import DATASET_ROOT
import os


def make_dataloader(dataset, base_size, crop_size, batch_size, num_workers, overfit, **kwargs):

    if dataset == 'cityscapes':
        dataset_path = os.path.join(DATASET_ROOT, dataset)
        train_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size, split='train', overfit=overfit)
        val_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size, split='val', overfit=overfit)
        num_classes = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs)
        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'active_cityscapes_image':
        dataset_path = os.path.join(DATASET_ROOT, 'cityscapes')

        train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                            split='train', init_set=kwargs['init_set'], overfit=overfit)
        val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                          split='val', init_set=kwargs['init_set'], overfit=overfit)
        num_classes = train_set.NUM_CLASSES

        del kwargs['init_set']
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,  **kwargs)

        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'active_cityscapes_region':
        dataset_path = os.path.join(DATASET_ROOT, 'cityscapes')

        train_set = region_cityscapes.ActiveCityscapesRegion(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                             split='train', init_set=kwargs['init_set'], overfit=overfit)
        val_set = region_cityscapes.ActiveCityscapesRegion(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                           split='val', init_set=kwargs['init_set'], overfit=overfit)
        num_classes = train_set.NUM_CLASSES

        del kwargs['init_set']
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,  **kwargs)

        return train_set, train_loader, val_loader, None, num_classes

    else:
        raise NotImplementedError
