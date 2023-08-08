
from img_color_gan.components.colorization_dataset import ColorizationDataset
from img_color_gan.components.data_transformation import DataTransformation
from img_color_gan import logger
import numpy as np
import torch

class DataLoader:
    def __init__(self, config, train_transform = None, val_transform = None, batch_size = 32):
        self.config = config
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.train_dataloader = None
        self.val_dataloader = None

    def get_data_loaders(self):
        train_paths = np.loadtxt(self.config.train_file, dtype = "str")
        val_paths = np.loadtxt(self.config.val_file, dtype = "str")

        logger.info("Getting train and validation data loaders.")
        train_dataset = ColorizationDataset(train_paths, self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size)

        val_dataset = ColorizationDataset(val_paths, self.val_transform)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size)

        return self.train_dataloader, self.val_dataloader
