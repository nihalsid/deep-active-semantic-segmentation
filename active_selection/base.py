class ActiveSelectionBase:

    def __init__(self, dataset_lmdb_env, crop_size, dataloader_batch_size):
        self.crop_size = crop_size
        self.dataloader_batch_size = dataloader_batch_size
        self.env = dataset_lmdb_env
