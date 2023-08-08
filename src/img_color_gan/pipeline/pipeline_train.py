from img_color_gan.config.configuration import ConfigurationManager
from img_color_gan.components.data_preparation import DataPreparation
from img_color_gan.components.data_loader import DataLoader
from img_color_gan.components.model import MainModel
from img_color_gan.components.model_trainer import ModelTrainer
from img_color_gan import logger
import glob
import mlflow

from img_color_gan.components.data_transformation import DataTransformation

STAGE_NAME = "Training stage"

class TrainPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        transform_config = config.get_transform_config()
        trainer_config = config.get_trainer_config()
        train_output_config = config.get_train_output_config()
        dataloader_config = config.get_data_loader_config()

        mlflow.log_params(transform_config.__dict__)
        mlflow.log_params(trainer_config.__dict__)

        data_transformation = DataTransformation(transform_config)
        train_transform = data_transformation.get_transform(split="train")
        val_transform = data_transformation.get_transform(split="val")
        
        dataloader = DataLoader(dataloader_config, train_transform, val_transform, trainer_config.batch_size)
        train_dataloader, val_dataloader = dataloader.get_data_loaders()

        model = MainModel(lr_G=trainer_config.lr_G, lr_D=trainer_config.lr_D, beta1=trainer_config.adam_beta1, beta2=trainer_config.adam_beta2, lambda_L1=trainer_config.lambda_L1)

        logger.info("Starting model training")
        logger.info(f"Using {model.device} for training.")
        model_trainer = ModelTrainer(trainer_config, model=model, train_dl=train_dataloader, val_dl=val_dataloader, train_output_config=train_output_config)
        model_trainer.train_model()

        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

