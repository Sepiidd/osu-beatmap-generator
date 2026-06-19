from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
import time

def eval_loss(model, train_loader, config):
    '''
    run test/evaluation time loss computation to see model accuracy
    '''
    #TODO
    validation_loader = config['validation_loader']
    ctx = config['ctx']
    eval_iters = config['eval_iters']
    criterion = config['criterion']

    model.eval()
    #evaluation, loss calc, accuracy calc goes here...
    accuracy = {}
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        loader = train_loader if split == train else validation_loader
        correct = 0
        total = 0
        for i, data in enumerate(loader):
            if i >= eval_iters:
                break

            inputs, targets = data
            with ctx:
                logits = model(inputs)
            loss = criterion(inputs, targets)

            losses[i] = loss.item()
            predictions = (logits >= 0.0)
            correct += (predictions == targets).sum().item()
            total += targets.shape[0]

        #average loss, determine accuracy
        out[split] = losses.mean()
        accuracy[split] = correct / total
            
    model.train()
    return out, accuracy

def write_checkpoints(epoch_idx, model, optimizer, best_val_loss, val_losses, config):
    #TODO
    pass

def train_one_epoch(epoch_idx, model, train_loader, optimizer, config):
    '''
    performs training (on gpu if possible) as well as the following additional steps + optimizations:
        -autocast, scaling 
            -look into <ctx> if confused
        -logging
        -model checkpointing
    '''
    #TODO: gradient accumulation, gradient clipping
    criterion = config['criterion']
    scaler = config['scaler']
    device = config['device']
    ctx = config['ctx']
    log_interval = config['log_interval']

    t0 = time.time()
    model.train()

    for i, data in enumerate(train_loader):
        inputs, targets = data
        break;

        #asynchronously move inputs and targets to gpu
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with ctx: #for autocast when working through gpu
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward() #backward pass
        scaler.step(optimizer) #optimizer step
        scaler.update() #dynamically updates magnitude/scale factor of scaler

        optimizer.zero_grad(set_to_none=True) #set as none over setting to zero (very minimal performance)

        #logging, loss evaluation, model snapshot related stuff
        t1 = time.time()
        dt = t1 - t0
        if i % log_interval == 0:
            print(f"iter {i}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
        if i % eval_interval == 0:
            losses, accuracy = eval_loss(model, train_loader, config)
            print(f"evaluation accuracies {i}: train loss {losses['train']:.4f} accuracy {accuracy['train']:.4f}, validation loss {losses['val']:.4f}, accuracy {accuracy['val']:.4f}")
#            write_checkpoints(model, optimizer, config)



def train_epochs(model, train_loader, config):
    #iterate <epochs> many times
    optimizer = model.configure_optimizer(config['weight_decay'], config['lr'], (config['beta1'], config['beta2']), config['device'])
    for epoch in range(config['epochs']):
        print(f"=====EPOCH NUMBER: {epoch}=====")
        train_one_epoch(epoch, model, train_loader, optimizer, config)


