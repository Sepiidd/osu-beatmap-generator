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
import os
import sys

BASE_DIR = Path(__file__).parent

if __name__ == "__main__":
    config = TrainingConfig()

    print("note: pass path to model state dict if training from existing model")
    is_fresh_model = True if len(sys.argv) < 2 else False

    model = OnsetModel(OnsetConfig()) #also sends to gpu if possible
    model = model.to(config.device)
    optimizer = model.configure_optimizer(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)

    starting_idx = 0
    if not is_fresh_model:
        model_path = sys.argv[1]
        try:
            model_path = Path(model_path).resolve()
            print("model path is", model_path)
            checkpoint = torch.load(model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_idx = checkpoint['iter_idx']
        except Exception as e:
            print("exception encountered:", e)
            exit(1)

    train(model, config.train_loader, optimizer, config, starting_idx)
