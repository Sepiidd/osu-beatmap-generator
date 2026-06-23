from dataclass.obg_audio_dataset import OBGAudioDataset
from models.onset.onset_model import OnsetModel
from models.onset.onset_config import OnsetConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from training.helpers_train import train_epochs
from training.helpers_train import train 
import torch.nn as nn
import torch
import easydict

BASE_DIR = Path(__file__).parent
SEQUENCE_LEN = 512

if __name__ == "__main__":
    #TODO
    train_h5path = BASE_DIR.parent.parent / "datasets" / "train"
    validation_h5path = BASE_DIR.parent.parent / "datasets" / "validation"
    test_h5path = BASE_DIR.parent.parent / "datasets" / "test"
    
    #check if gradscaler is required
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=pt_dtype)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == 'float16')) #if not working in float16, acts as a no-op

    train_dataset = OBGAudioDataset(train_h5path, SEQUENCE_LEN)
    validation_dataset = OBGAudioDataset(validation_h5path, SEQUENCE_LEN)
    test_dataset = OBGAudioDataset(test_h5path, SEQUENCE_LEN)

    trainloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            pin_memory=True #for faster and async gpu transfers
            )
    validationloader = DataLoader(
            validation_dataset,
            batch_size=8,
            shuffle=True,
            pin_memory=True #for faster and async gpu transfers
            )
    testloader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=True,
            pin_memory=True #for faster and async gpu transfers
            )

    free_mem, total_mem = torch.cuda.mem_get_info()
    print("gpu free_mem and total_mem are", free_mem, total_mem)

    criterion = nn.BCEWithLogitsLoss()
    model = OnsetModel(OnsetConfig()).to(device) #also sends to gpu if possible

    train_config = {
        'epochs': 1,
        'max_iters': 100, #set to specific value if not passing over entire dataset per epoch
        'log_interval': 10,
        'eval_interval': 10,
        'eval_iters': 40,
        'device': device,
        'ctx': ctx,
        'train_loader': trainloader,
        'validation_loader': validationloader,
        'test_loader': testloader,
        'criterion': criterion,
        'scaler': scaler,
        'weight_decay': 0.1,
        'lr': 3e-5,
        'beta1': 0.9,
        'beta2': 0.99,
        'grad_clip': 1.0,
        'grad_accumulation_steps': 4
    }
    config = easydict.EasyDict(train_config)

    print("config is", config)

    optimizer = model.configure_optimizer(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    train(model, trainloader, optimizer, config)
#    train_epochs(model, trainloader, easydict.EasyDict(train_config))
