from dataloaders.dataset import cityscapes, active_cityscapes, region_cityscapes, pascal, active_pascal, region_pascal
from torch.utils.data import DataLoader
from constants import DATASET_ROOT
import os


def make_dataloader(dataset, base_size, crop_size, batch_size, num_workers, overfit, **kwargs):

    if dataset == 'cityscapes':
        dataset_path = os.path.join(DATASET_ROOT, dataset)
        train_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                          split='train', overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        val_set = cityscapes.Cityscapes(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                        split='val', overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        del kwargs['memory_hog']
        num_classes = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs)
        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'active_cityscapes_image':
        dataset_path = os.path.join(DATASET_ROOT, 'cityscapes')

        train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                            split='train', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        val_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                          split='val', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        num_classes = train_set.NUM_CLASSES

        del kwargs['init_set']
        del kwargs['memory_hog']
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,  **kwargs)

        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'active_cityscapes_region':
        dataset_path = os.path.join(DATASET_ROOT, 'cityscapes')

        train_set = region_cityscapes.ActiveCityscapesRegion(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                             split='train', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        val_set = region_cityscapes.ActiveCityscapesRegion(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                           split='val', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        num_classes = train_set.NUM_CLASSES
        del kwargs['memory_hog']
        del kwargs['init_set']
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,  **kwargs)

        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'pascal':
        dataset_path = os.path.join(DATASET_ROOT, 'pascal')
        train_set = pascal.Pascal(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                  split='train', overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        val_set = pascal.Pascal(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                split='val', overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        del kwargs['memory_hog']
        num_classes = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs)
        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'active_pascal_image':
        dataset_path = os.path.join(DATASET_ROOT, 'pascal')

        train_set = active_pascal.ActivePascalImage(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                    split='train', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        val_set = active_pascal.ActivePascalImage(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                  split='val', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        num_classes = train_set.NUM_CLASSES

        del kwargs['init_set']
        del kwargs['memory_hog']
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,  **kwargs)

        return train_set, train_loader, val_loader, None, num_classes

    if dataset == 'active_pascal_region':
        dataset_path = os.path.join(DATASET_ROOT, 'pascal')

        train_set = region_pascal.ActivePascalRegion(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                     split='train', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        val_set = region_pascal.ActivePascalRegion(path=dataset_path, base_size=base_size, crop_size=crop_size,
                                                   split='val', init_set=kwargs['init_set'], overfit=overfit, memory_hog_mode=kwargs['memory_hog'])
        num_classes = train_set.NUM_CLASSES
        del kwargs['memory_hog']
        del kwargs['init_set']
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,  **kwargs)

        return train_set, train_loader, val_loader, None, num_classes

    else:
        raise NotImplementedError
