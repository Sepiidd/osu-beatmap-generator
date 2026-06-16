from dataclass.obg_audio_dataset import OBGAudioDataset
from models.onset.onset_model import OnsetModel
from models.onset.onset_config import OnsetConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from training.helpers_train import train_epochs
import torch.nn as nn
import torch

BASE_DIR = Path(__file__).parent
SEQUENCE_LEN = 512

if __name__ == "__main__":
    print("cuda is available:", torch.cuda.is_available())
    h5path = BASE_DIR.parent.parent / "datasets" / "partition0"
    dataset = OBGAudioDataset(h5path, SEQUENCE_LEN)
    dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True
            )
    model = OnsetModel(OnsetConfig())
    criterion = nn.BCEWithLogitsLoss()

    #training configuration
    train_config = {
        'epochs': 40
        'criterion': criterion
    }

    train_epochs(model, dataloader, train_config)
