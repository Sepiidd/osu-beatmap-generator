from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch

def eval_loss():
    #TODO
    pass

def write_checkpoints():
    #TODO
    pass

def train_one_epoch(epoch_idx, model, train_loader, optimizer, config):
    #TODO
    criterion = config['criterion']
    scaler = config['scaler']
    device = config['device']
    ctx = config['ctx']

    model.train()
    for i, data in enumerate(train_loader):
        inputs, targets = data
        break; #TODO: loss tracking

        #asynchronously move inputs and targets to gpu
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with ctx: #for autocast when working on gpu
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward() #backward pass
        scaler.step(optimizer) #optimizer step
        scaler.update() #dynamically updates magnitude/scale factor of scaler

        optimizer.zero_grad(set_to_none=True) #set as none over setting to zero (very minimal performance)

def train_epochs(model, train_loader, validation_loader, config):
    #TODO
    #iterate <epochs> many times
    optimizer = model.configure_optimizer(config['weight_decay'], config['lr'], (config['beta1'], config['beta2']), config['device'])
    for epoch in range(config['epochs']):
        print(f"EPOCH NUMBER: {epoch}")
        train_one_epoch(epoch, model, train_loader, optimizer, config)

    #   after each epoch:
    #       perform validation
    #       snapshot model params/model state


