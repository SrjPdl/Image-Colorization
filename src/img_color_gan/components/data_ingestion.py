import os
import urllib.request as request
import zipfile
import tarfile
from img_color_gan import logger
from img_color_gan.utils.common import get_size
from pathlib import Path
from img_color_gan.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    
    def extract_tar_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with tarfile.open(self.config.local_data_file, 'r') as tar_ref:
            tar_ref.extractall(unzip_path)
  