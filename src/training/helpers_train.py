from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
import time

BASE_DIR = Path(__file__).parent

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

def write_checkpoints(epoch_idx, iter_idx, model, optimizer, val_loss_best, val_acc_best, config, target="onset_checkpoints"):
    path = BASE_DIR.parent.parent / target / f"e{epoch_idx}_i{iter_idx}_bl{val_loss_best}_ba{val_acc_best}"
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()

    obj = {
        'epoch_idx': epoch_idx,
        'iter_idx': iter_idx,
        'model_state_dict': model_dict,
        'optimizer_state_dict': opt_dict,
        'loss_best': val_loss_best,
        'acc_best': val_acc_best
    }
    torch.save(obj, path)

def train_one_epoch(epoch_idx, model, train_loader, optimizer, config):
    '''
    performs training (on gpu if possible) as well as the following additional steps + optimizations:
        -autocast, scaling 
            -look into <ctx> if confused
        -logging
        -model checkpointing
        -gradient clipping
    '''
    #TODO: gradient accumulation necessary?, randomize weight initialization (in onset model class)
    criterion = config['criterion']
    scaler = config['scaler']
    device = config['device']
    ctx = config['ctx']
    log_interval = config['log_interval']
    grad_clip = config['grad_clip']

    t0 = time.time()
    model.train()
    val_best = float('inf')
    val_acc_best = float('inf')

    for i, data in enumerate(train_loader):
        inputs, targets = data
        break; #NOTE

        #asynchronously move inputs and targets to gpu
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with ctx: #for autocast when working through gpu
            logits = model(inputs)
            loss = criterion(logits, targets)

        #grad clip
        if grad_clip != 0.0:
            scaler.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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
            val_loss = losses['val']
            val_acc = accuracy['val']
            val_best = val_loss if val_loss < val_best else val_best
            val_acc_best = val_acc if val_acc < val_acc_best else val_acc_best
            print(f"evaluation accuracies {i}: train loss {losses['train']:.4f} accuracy {accuracy['train']:.4f}, validation loss {losses['val']:.4f}, accuracy {accuracy['val']:.4f}")
            write_checkpoints(epoch_idx, i, model, optimizer, val_acc, val_acc_best, config)



def train_epochs(model, train_loader, config):
    #iterate <epochs> many times
    optimizer = model.configure_optimizer(config['weight_decay'], config['lr'], (config['beta1'], config['beta2']), config['device'])
    for epoch in range(config['epochs']):
        print("\n")
        print(f"=====EPOCH NUMBER: {epoch}=====")
        train_one_epoch(epoch, model, train_loader, optimizer, config)


