from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
import time
from datetime import datetime

BASE_DIR = Path(__file__).parent

def eval_loss(model, train_loader, config):
    '''
    run test/evaluation time loss computation to see model accuracy
    '''
    validation_loader = config.validation_loader
    ctx = config.ctx
    eval_iters = config.eval_iters
    criterion = config.criterion

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
            predictions = (logits >= 0.0) #threshold for logits is 0.0, i.e. 50% after softmax
            correct += (predictions == targets).sum().item()
            total += targets.shape[0]

        #average loss, determine accuracy
        out[split] = losses.mean()
        accuracy[split] = correct / total
            
    model.train()
    return out, accuracy

def write_checkpoints(iter_idx, model, optimizer, val_loss_best, val_acc_best, config, target="onset_checkpoints"):
    path = BASE_DIR.parent.parent / target / f"i{iter_idx}_bl{val_loss_best}_ba{val_acc_best}_date{datetime.now().strftime('%Y/%m/%d|%H:%M:%S')}"
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()

    obj = {
        'iter_idx': iter_idx,
        'model_state_dict': model_dict,
        'optimizer_state_dict': opt_dict,
        'loss_best': val_loss_best,
        'acc_best': val_acc_best
    }
    torch.save(obj, path)

def make_generator(dataloader, device):
    '''
    returns a single batch from <dataloader>, automatically reshuffling the batch when exhausted
    '''
    while True:
        for inputs, targets in dataloader:
            #asynchronously move inputs and targets to gpu
            i = inputs.to(device, non_blocking=True)
            t = targets.to(device, non_blocking=True)
            yield i, t

def get_lr(it, config):
    '''
    copied over from school reference
    '''
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters) #distance from warmup iters / total distance to decay
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # follows cosine shaped curve with range [0,1]
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def train(model, train_loader, optimizer, config):
    '''
    performs training (on gpu if possible) as well as the following additional steps + optimizations:
        -autocast, scaling 
            -look into <ctx> if confused
        -logging
        -model checkpointing
        -gradient clipping
        -gradient accumulation
    '''
    criterion = config.criterion
    scaler = config.scaler
    device = config.device
    ctx = config.ctx
    max_iters = config.max_iters
    log_interval = config.log_interval
    eval_interval = config.eval_iters
    grad_clip = config.grad_clip
    grad_accumulation_steps = config.grad_accumulation_steps

    t0 = time.time()
    model.train()
    val_best = float('inf')
    val_acc_best = 0.0

    train_gen = make_generator(train_loader, device)
    idx = 0
    inputs, targets = next(train_gen) #first batch

    while True:
        #TODO: variable learning rate (startup lr, etc), debug
        break

        if idx >= max_iters:
            break

        #variable learning rate
        lr = get_lr(idx, config) if config.decay_lr else config.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #predictions, back pass
        for microstep in range(grad_accumulation_steps):
            with ctx: #autocast type gpu work 
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss = loss / grad_accumulation_steps #manipulate algebra, simulate larger batch
            inputs, targets = next(train_gen)
            scaler.scale(loss).backward()
       #gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer) #optimizer step
        scaler.update() #dynamically updates magnitude/scale factor of scaler
        optimizer.zero_grad(set_to_none=True) #set as none over setting to zero (performance)

        #logging, loss evaluation, model snapshot related stuff
        t1 = time.time()
        dt = t1 - t0
        if idx % log_interval == 0:
            loss_scaled = loss.item() * grad_accumulation_steps
            print(f"iter {idx}: loss {loss_scaled:.4f}, time {dt*1000:.2f}ms")
        if idx % eval_interval == 0:
            losses, accuracy = eval_loss(model, train_loader, config)
            val_loss = losses['val']
            val_acc = accuracy['val']
            val_best = val_loss if val_loss < val_best else val_best
            val_acc_best = val_acc if val_acc > val_acc_best else val_acc_best
            print(f"evaluation accuracies {idx}: train loss {losses['train']:.4f} accuracy {accuracy['train']:.4f}, validation loss {losses['val']:.4f}, accuracy {accuracy['val']:.4f}")
            write_checkpoints(idx, model, optimizer, val_acc, val_acc_best, config)
        idx+=1


#NOTE: NO LONGER NEEDED
def train_epochs(model, train_loader, config):
    #iterate <epochs> many times
    optimizer = model.configure_optimizer(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    for epoch in range(config.epochs):
        print("\n")
        print(f"=====EPOCH NUMBER: {epoch}=====")
        train_one_epoch(epoch, model, train_loader, optimizer, config)


