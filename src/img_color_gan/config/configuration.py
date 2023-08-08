from img_color_gan.constants import *
from img_color_gan.utils.common import read_yaml, create_directories
from img_color_gan.entity.config_entity import (DataIngestionConfig,
                                                DataTransformConfig,
                                                DataPreparationConfig,
                                                DataLoaderConfig,
                                                TrainerConfig,
                                                TrainOutputConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation

        create_directories([config.prepare_path])

        data_preparation_config = DataPreparationConfig(
             data_dir = config.data_dir,
             num_images_to_use = config.num_images_to_use,
             val_size = config.val_size,
             seed = config.seed,
             prepare_path = config.prepare_path
        )

        return data_preparation_config
    
    
    def get_transform_config(self) -> DataTransformConfig:
        trnsfrms = self.config.data_transformation
        trnsfrms_config = DataTransformConfig(
            trnsfrms.size,
            trnsfrms.random_horizontal_flip_p,
            trnsfrms.random_vertical_flip_p,
            trnsfrms.random_rotation_range
        )
        return trnsfrms_config
    
    def get_data_loader_config(self) -> DataLoaderConfig:
        config = self.config.data_loader
        data_loader_config = DataLoaderConfig(
            config.train_file,
            config.val_file
        )
        return data_loader_config
    
    def get_trainer_config(self) -> TrainerConfig:
        config = self.params.train_params

        train_params = TrainerConfig(
            config.num_epochs,
            config.batch_size,
            config.lr_G,
            config.lr_D,
            config.adam_beta1,
            config.adam_beta2,
            config.lambda_L1
        )
        return train_params
    
    def get_train_output_config(self) -> TrainOutputConfig:
        config = self.config.train_outputs
        train_output_config = TrainOutputConfig(
            config.model_dir,
            config.sample_image_dir,
            config.save_img_freq
        )
        return train_output_config
