
from img_color_gan.utils.common import create_loss_meters, update_losses, log_results, visualize, save_ckpt, load_ckpt
from img_color_gan import logger
from tqdm import tqdm
from pathlib import Path

class ModelTrainer:
    def __init__(self, config, model, train_dl, val_dl, train_output_config):
        self.config = config
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_output_config = train_output_config
    
    def train_model(self):
        data = next(iter(self.val_dl)) # getting a batch for visualizing the model output after fixed intrvals
        for e in range(self.config.num_epochs):
            loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
            i = 0                                  # log the losses of the complete network
            for data in tqdm(self.train_dl):
                self.model.setup_input(data) 
                self.model.optimize()
                update_losses(self.model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
                i += 1
                if i % self.train_output_config.save_img_freq == 0:
                    logger.info(f"\nEpoch {e+1}/{self.config.num_epochs}")
                    logger.info(f"Iteration {i}/{len(self.train_dl)}")
                    log_results(loss_meter_dict) # function to print out the losses
                    visualize(self.model, data, epoch = e, iteration = i, save_path=self.train_output_config.sample_image_dir) # function displaying the model's outputs
        logger.info(f"Saving model at {self.train_output_config.model_dir}")
        save_ckpt(e, self.model, self.train_output_config.model_dir)