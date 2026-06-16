from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch

def train_one_epoch(epoch_idx, model, loader, config):
    iter_num = 0
    for i, data in enumerate(loader):
        inputs, targets = data
        print("inputs has shape", inputs.shape, "with size (bytes)", inputs.nbytes)
        print("targets has shape", targets.shape, "with size (bytes)", inputs.nbytes)
        break;
        #TODO: training stuff

def train_epochs(model, loader, config):
    #iterate <epochs> many times
    for _ in range(epochs):
        #TODO
        pass
    #   after each epoch:
    #       perform validation
    #       snapshot model params/model state
