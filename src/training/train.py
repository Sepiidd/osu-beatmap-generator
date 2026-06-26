from dataclass.obg_audio_dataset import OBGAudioDataset
from models.onset.onset_model import OnsetModel
from configs.onset_config import OnsetConfig
from configs.training_config import TrainingConfig
from pathlib import Path
from torch.utils.data import DataLoader
from training.helpers_train import train 
import torch.nn as nn
import torch
import easydict


BASE_DIR = Path(__file__).parent

if __name__ == "__main__":
    config = TrainingConfig()

    free_mem, total_mem = torch.cuda.mem_get_info()
    print("gpu free_mem and total_mem are", free_mem, total_mem)

    model = OnsetModel(OnsetConfig()).to(config.device) #also sends to gpu if possible
    print("config is", config)

    optimizer = model.configure_optimizer(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    train(model, config.train_loader, optimizer, config)
