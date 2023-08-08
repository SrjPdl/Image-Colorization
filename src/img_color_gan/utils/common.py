import os
from box.exceptions import BoxValueError
import yaml
from img_color_gan import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import torch
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import matplotlib.pyplot as plt
import time
import mlflow


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        if verbose:
            if not os.path.exists(path):
                logger.info(f"created directory at: {path}")
            else:
                logger.info(f"Using existing directory {path}")
        os.makedirs(path, exist_ok=True)


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data, epoch, iteration, save_path):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    image_dir = f"{save_path}/{epoch}"
    create_directories([image_dir])
    logger.info(f"Saving images [epoch/iter: {epoch}/{iteration}]")
    image_path = f"{image_dir}/{iteration}_colorization_{time.time()}.png"
    fig.savefig(image_path)
    mlflow.log_artifact(image_path, artifact_path=f"images/{epoch}")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        mlflow.log_metric(loss_name, loss_meter.avg)
        logger.info(f"{loss_name}: {loss_meter.avg:.5f}")


def save_ckpt(epoch, model, save_dir):
    """Saves the checkpoint of the model.
    Args:
        epoch (int): The current epoch.
        model (torch.nn.Module): The model to save.
        save_dir (str): Directory to save the checkpoint.

    Returns:
        None
    """
    ckpt = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
    os.makedirs(save_dir, exist_ok = True)
    torch.save(ckpt, f"{save_dir}/model_ckpt_{time.time()}.pt")

def load_ckpt(ckpt, model):
    """Loads the checkpoint of the model.

    Args:
        ckpt (str): Path to the checkpoint file.
        num_classes(int): Number of classes present in images.

    Returns:
     tuple: A tuple containing the loaded model, and epoch trained.
    """
        
    ckpt_loaded = torch.load(ckpt)
    model.load_state_dict(ckpt_loaded['model_state_dict'])
    epoch_trained = ckpt_loaded['epoch']
    return model, epoch_trained