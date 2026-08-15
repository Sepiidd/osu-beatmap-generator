from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
from configs.gen_config import GenConfig
from torchmetrics.classification import BinaryAveragePrecision
import os
import traceback
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import math
import sys

BASE_DIR = Path(__file__).parent
configG = GenConfig()

@torch.no_grad()
def eval_loss(model, train_loader, validation_loader, config):
    '''
    run test/evaluation time loss computation to see model accuracy
    '''
    ctx = config.ctx
    eval_iters = config.eval_iters
    criterion = config.criterion

    model.eval()
    #evaluation, loss calc, accuracy calc goes here...
    stats = {}
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        loader = train_loader if split == "train" else validation_loader
        
        #statistics
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0
        aucpr_calc = BinaryAveragePrecision()

        i=0
        while True:
            if i >= eval_iters:
                break

            inputs, difficulty, targets = next(loader)
            with ctx:
                logits = model(inputs, difficulty).squeeze()
                targets = targets.squeeze().to(config.pt_dtype) #match type of logits
                loss = criterion(logits, targets)
            losses[i] = loss.item()
            predictions = F.sigmoid(logits)

            aucpr_calc.update(predictions, targets.to(int))

            predictions = predictions >= configG.prediction_threshold 

            batch_tp = (predictions * targets).sum()
            tp += batch_tp
            batch_fp = (predictions * (1-targets)).sum()
            fp += batch_fp
            batch_fn = ((~predictions) * targets).sum()
            fn += batch_fn
            
            batch_correct = (predictions == targets).sum().item()
            correct += batch_correct 

            if len(targets.shape) == 1:
                total += targets.shape[0] # sequence length
            else:
                total += targets.shape[0]*targets.shape[1] #batch size * sequence length
            i+=1

        #average loss, determine stats 
        precision = (tp / (tp+fp + 1e-7)).item() #of all positive predictions, which were correct
        recall = (tp / (tp+fn + 1e-7)).item()
        f_score = ((2*tp) / (2*tp+fp+fn + 1e-7)).item()

        aucpr = (aucpr_calc.compute()).item()
        aucpr_calc.reset()

        out[split] = losses.mean()
        stats[split] = {"precision": precision, "recall": recall, "accuracy": correct / total, "f_score": f_score, "aucpr": aucpr}
    model.train()
    return out, stats 

def write_checkpoints(iter_idx, model, optimizer, stats, config):
    path = Path(config.checkpoint_path) / f"iter{iter_idx}_tauc{stats['train_aucpr']:.4f}vauc{stats['val_aucpr']:.4f}_tf{stats['train_f_score']}vf{stats['val_f_score']}_date{datetime.now().strftime('%Y-%m-%d|%H:%M:%S')}"
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()

    obj = {
        'iter_idx': iter_idx,
        'model_state_dict': model_dict,
        'optimizer_state_dict': opt_dict,
        "train_accuracy": stats['train_accuracy'],
        "train_precision": stats['train_precision'],
        "train_recall": stats['train_recall'],
        "train_aucpr": stats['train_aucpr'],
        "train_aucpr_best": stats['train_aucpr_best'],
        "train_f_score": stats['train_f_score'],
        "train_f_score_best": stats['train_f_score_best'],
        "train_loss": stats['train_loss'],
        "train_loss_best": stats['train_loss_best'],
        "val_accuracy": stats['val_accuracy'],
        "val_precision": stats['val_precision'],
        "val_recall": stats['val_recall'],
        "val_aucpr": stats['val_aucpr'],
        "val_aucpr_best": stats['val_aucpr_best'],
        "val_f_score": stats['val_f_score'],
        "val_f_score_best": stats['val_f_score_best'],
        "val_loss": stats['val_loss'],
        "val_loss_best": stats['val_loss_best'],
    }
    compare_save_checkpoint(config.checkpoint_path, obj, path)
#    torch.save(obj, path)

def compare_save_checkpoint(checkpoint_dir, checkpoint, save_path, max_checkpoints=5):
    '''
    compares the given checkpoint <checkpoint> stats to existing checkpoints in <checkpoint_dir> to determine if <checkpoint> should be saved to disk
    '''
    num_checkpts = 0
    idx = 0
    checkpoints = []
    min_checkpoint_idx = -1
    curr_valaucpr = checkpoint['val_aucpr']
    for f in checkpoint_dir.iterdir():
        if not f.name.startswith("iter"):
            continue
        checkpoints.append(f)
        num_checkpts+=1

        c = torch.load(f)
        val_aucpr = c['val_aucpr']
        if val_aucpr < curr_valaucpr:
            curr_valaucpr = val_aucpr
            min_checkpoint_idx = idx
        idx+=1

    if min_checkpoint_idx == -1 and num_checkpts > 0: #checkpoint performs worse than all existing checkpoints
        return
    if num_checkpts >= max_checkpoints:
        rm = checkpoints[min_checkpoint_idx]
        rm.unlink()
    #min_checkpoint_idx == -1 equivalent to num_checkpts<=0, save obj in these cases
    print("saving checkpoint")
    torch.save(checkpoint, save_path)


def make_generator(dataloader, device):
    '''
    returns a single batch from <dataloader>, automatically reshuffling the batch when exhausted
    '''
    while True:
        for inputs, difficulty, targets in dataloader:
            #asynchronously move inputs and targets to gpu
            i = inputs.to(device, non_blocking=True)
            d = difficulty.to(device, non_blocking=True)
            t = targets.to(device, non_blocking=True)
            yield i, d, t

def get_lr(it, config):
    '''
    copied over from school reference
    '''
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.lr * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters) #distance from warmup iters / total distance to decay
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # follows cosine shaped curve with range [0,1]
    return config.min_lr + coeff * (config.lr - config.min_lr)

def train(model, train_loader, optimizer, config, starting_idx=0):
    '''
    performs training (on gpu if possible) as well as the following additional steps + optimizations:
        -autocast, scaling 
            -look into <ctx> if confused
        -logging
        -model checkpointing
        -gradient clipping
        -gradient accumulation
    '''
    print("beginning training loop...")

    #config
    criterion = config.criterion
    scaler = config.scaler
    device = config.device
    ctx = config.ctx
    max_iters = config.max_iters
    log_interval = config.log_interval
    eval_interval = config.eval_iters
    checkpoint_iters = config.checkpoint_iters
    grad_clip = config.grad_clip
    grad_accumulation_steps = config.grad_accumulation_steps

    #stats
    val_loss_best = float('inf')
    train_loss_best = float('inf')
    train_f_score_best = 0.0
    val_f_score_best = 0.0
    train_aucpr_best = 0.0
    val_aucpr_best = 0.0
    t0 = time.time()
    track_train_aucpr = []
    track_train_fscore = []
    track_val_aucpr = []
    track_val_fscore = []
    track_time = []

    #generators
    train_gen = make_generator(train_loader, device)
    eval_train_gen = make_generator(train_loader, device)
    eval_val_gen = make_generator(config.validation_loader, device)
    idx = starting_idx
    inputs, difficulty, targets = next(train_gen) #first batch
    
    model.train()
    try:
        while True:
            if max_iters is not None and idx >= max_iters:
                break

            #variable learning rate
            lr = get_lr(idx, config) if config.decay_lr else config.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            #predictions, back pass
            for microstep in range(grad_accumulation_steps):
                with ctx: #autocast type gpu work 

                    logits = model(inputs, difficulty).squeeze()
                    targets = targets.squeeze().to(config.pt_dtype) #match type of logits
                    loss = criterion(logits, targets)
                    loss = loss / grad_accumulation_steps #manipulate algebra, simulate larger batch
                inputs, difficulty, targets = next(train_gen)
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
                print(f"iter {idx}: loss {loss_scaled:.4f}, time {dt*1000:.2f}ms, curr lr {lr}")
            if idx % eval_interval == 0:
                losses, stats = eval_loss(model, eval_train_gen, eval_val_gen, config)

                train_loss = losses['train']
                train_loss_best = train_loss if train_loss < train_loss_best else train_loss_best
                val_loss = losses['val']
                val_loss_best = val_loss if val_loss < val_loss_best else val_loss_best

                train_f_score = stats['train']['f_score']
                train_f_score_best = train_f_score if train_f_score > train_f_score_best else train_f_score_best
                val_f_score = stats['val']['f_score']
                val_f_score_best = val_f_score if val_f_score > val_f_score_best else val_f_score_best

                train_aucpr = stats['train']['aucpr']
                train_aucpr_best = train_aucpr if train_aucpr > train_aucpr_best else train_aucpr_best
                val_aucpr = stats['val']['aucpr']
                val_aucpr_best = val_aucpr if val_aucpr > val_aucpr_best else val_aucpr_best
                

                stats_checkpoint = {
                    "train_accuracy": stats['train']['accuracy'],
                    "train_precision": stats['train']['precision'],
                    "train_recall": stats['train']['recall'],
                    "train_aucpr": stats['train']['aucpr'],
                    "train_aucpr_best": train_aucpr_best,
                    "train_f_score": train_f_score,
                    "train_f_score_best": train_f_score_best,
                    "train_loss": train_loss,
                    "train_loss_best": train_loss_best,
                    "val_accuracy": stats['val']['accuracy'],
                    "val_precision": stats['val']['precision'],
                    "val_recall": stats['val']['recall'],
                    "val_aucpr": stats['val']['aucpr'],
                    "val_aucpr_best": val_aucpr_best,
                    "val_f_score": val_f_score,
                    "val_f_score_best": val_f_score_best,
                    "val_loss": val_loss,
                    "val_loss_best": val_loss_best,
                }
                
                track_train_aucpr.append(stats_checkpoint["train_aucpr"])
                track_val_aucpr.append(stats_checkpoint["val_aucpr"])
                track_train_fscore.append(stats_checkpoint["train_f_score"])
                track_val_fscore.append(stats_checkpoint["val_f_score"])
                track_time.append(dt)

                print(f"evaluation step {idx} - loss: train {losses['train']:.4f}, val {losses['val']:.4f} | f-score: train {train_f_score} val {val_f_score} | aucpr: train {stats['train']['aucpr']} val {stats['val']['aucpr']} gap {stats['train']['aucpr']-stats['val']['aucpr']} | precision: train {stats['train']['precision']} val {stats['val']['precision']} | recall: train {stats['train']['recall']} val {stats['val']['recall']}")  

                if idx % checkpoint_iters == 0:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    print(f"writing checkpoints and memory check | gpu free_mem: {free_mem}, total_mem: {total_mem}")
                    write_checkpoints(idx, model, optimizer, stats_checkpoint, config)

                    if val_aucpr >= config.aucpr_goal:
                        print("\n==========VALIDATION SET AUC-PR GOAL ACHIEVED, STOPPING TRAINING==========")
                        break
            idx+=1
    except KeyboardInterrupt:
        print("\n==========Manual Early Stop Performed==========")
    except Exception as e:
        print("==========Exception Occurred (error.txt)==========\n")
        traceback.print_exception(e)
    finally:
        #plot auc pr and f score over time
        print("plotting auc-pr and f-score over time")
        plot_thing_over_time(track_train_aucpr, track_val_aucpr, track_time, "aucpr")
        plt.figure()
        plot_thing_over_time(track_train_fscore, track_val_fscore, track_time, "fscore")

def plot_thing_over_time(aucpr_t, aucpr_v, time, filename):
    plt.plot(aucpr_t, alpha=0.7)
    plt.plot(aucpr_v, alpha=0.7)
    plt.xlabel(f"Time")
    plt.ylabel("AUC-PR")
    plt.ylim(0,1)
    plt.title("AUC-PR Over Time")
    plt.savefig(filename + ".png")
