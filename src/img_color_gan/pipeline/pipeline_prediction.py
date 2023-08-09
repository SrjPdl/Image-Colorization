from img_color_gan.config.configuration import ConfigurationManager
from img_color_gan.components.data_transformation import DataTransformation
from img_color_gan.utils.common import load_ckpt, lab_to_rgb
from img_color_gan import logger
from img_color_gan.components.model import MainModel
from PIL import Image
import torch
import numpy as np
from skimage.color import rgb2lab
from torchvision import transforms


class PredictionPipeline:
    def __init__(self, ckpt_path):
        self.config = ConfigurationManager()
        data_transform_config = self.config.get_transform_config()
        data_transform = DataTransformation(config = data_transform_config)
        self.transform = data_transform.get_transform(split = "test")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MainModel()
        load_ckpt(ckpt_path, self.model, map_location=self.device)
        self.model = self.model.to(self.device)

    def predict(self, image):
        image = self.transform(image)
        image = np.array(image)
        image_lab = rgb2lab(image).astype("float32")
        image_lab = transforms.ToTensor()(image_lab)
        L = image_lab[[0], ...] / 50. - 1.
        L = torch.unsqueeze(L, 0).to(self.device)
        self.model.net_G.eval()
        with torch.no_grad():
            fake_color =self.model.net_G(L)

        fake_img = lab_to_rgb(L, fake_color)
        return fake_img[0]


