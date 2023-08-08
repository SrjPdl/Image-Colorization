from img_color_gan.entity.config_entity import DataTransformConfig
from img_color_gan import logger
from torchvision import transforms
from PIL import Image

class DataTransformation:
    def __init__(self, config: DataTransformConfig):
        self.config = config
    
    def get_transform(self, split = "train"):
        logger.info(f"Getting {split} transform.")
        if split == "train":
            transform = transforms.Compose([
                transforms.Resize((self.config.size, self.config.size),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(p = self.config.random_horizontal_flip_p),
                transforms.RandomVerticalFlip(p = self.config.random_horizontal_flip_p),
                transforms.RandomRotation(self.config.random_rotation_range)
                ])
            return transform
        elif split == "val":
            transform = transforms.Compose([
                transforms.Resize((self.config.size, self.config.size),  Image.BICUBIC)
                ])
            return transform

