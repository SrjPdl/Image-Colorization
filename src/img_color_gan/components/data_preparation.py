from img_color_gan.entity.config_entity import DataPreparationConfig
from img_color_gan import logger
import os
import glob
import numpy as np

class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
    
    def prepare_data(self):
        data_path = self.config.data_dir
        seed = self.config.seed
        num_images_to_use = self.config.num_images_to_use
        val_size = self.config.val_size
        prepare_path = self.config.prepare_path
        num_train_images = int(num_images_to_use*(1-val_size))

        logger.info("Splitting data into train and val set.")
        paths = glob.glob(data_path + "/*.jpg") # Grabbing all the image file names
        np.random.seed(seed)
        paths_subset = np.random.choice(paths, num_images_to_use, replace=False) # choosing 1000 images randomly
        rand_idxs = np.random.permutation(num_images_to_use)
        train_idxs = rand_idxs[:num_train_images] # choosing the first 8000 as training set
        val_idxs = rand_idxs[num_train_images:] # choosing last 2000 as validation set
        train_paths = paths_subset[train_idxs]
        val_paths = paths_subset[val_idxs]
        logger.info(f"Train Images: {len(train_paths)} Val Images: {len(val_paths)}")

        train_file = os.path.join(prepare_path, "train.txt")
        if not os.path.exists(train_file):
            logger.info(f"Writing train image paths {train_file}.")
            np.savetxt(train_file, train_paths, fmt='%s')
        else:
            logger.info("Train file already exists.")

        val_file = os.path.join(prepare_path, "val.txt")
        if not os.path.exists(val_file):
            logger.info(f"Writing val image paths {val_file}.")
            np.savetxt(val_file, val_paths, fmt='%s')
        else:
            logger.info("Validation file already exists.")

