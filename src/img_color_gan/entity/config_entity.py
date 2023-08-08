from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    data_dir: str
    num_images_to_use: int
    val_size: float
    seed: int
    prepare_path: str


@dataclass(frozen=True)
class DataTransformConfig:
    size: int
    random_horizontal_flip_p: float
    random_vertical_flip_p: float
    random_rotation_range: float


@dataclass(frozen=True)
class DataLoaderConfig:
    train_file: str
    val_file: str

@dataclass(frozen=True)
class TrainerConfig:
    num_epochs: int
    batch_size: int
    lr_G: float
    lr_D: float
    adam_beta1: float
    adam_beta2: float
    lambda_L1: float

@dataclass(frozen=True)
class TrainOutputConfig:
    model_dir: str
    sample_image_dir: str
    save_img_freq: int