from dataclasses import dataclass
from dataclass.obg_audio_dataset import OBGAudioDataset
from torch.utils.data import DataLoader
from pathlib import Path
from configs.audio_config import AudioConfig
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).parent
SEQ_LEN = AudioConfig().sequence_len

#not dataclass requires more logic
class TrainingConfig():
    def __init__(self):
        #==========STATIC VALUES==========
        #training iteration breakpoints
        self.sequence_len = SEQ_LEN
        self.max_iters = 10000
        self.log_interval = 10
        self.eval_interval = 10
        self.eval_iters = 40
        self.warmup_iters = 1000 #experiment with this and decay_iters
        self.lr_decay_iters = 5000
        #model training specifics
        self.weight_decay = 0.1
        self.lr = 3e-4
        self.min_lr = 3e-5
        self.decay_lr = False
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.grad_clip = 1.0
        self.grad_accumulation_steps = 4
        #dataloaders
        self.train_path = BASE_DIR.parent.parent / "datasets" / "train"
        self.validation_path = BASE_DIR.parent.parent / "datasets" / "validation" 
        self.test_path = BASE_DIR.parent.parent / "datasets" / "test"
        self.batch_size = 8
        self.shuffle = True
        self.pin_memory = True

        #==========COMPUTED VALUES==========
        #check if cuda is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #check if gradscaler is required
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        self.pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=self.pt_dtype)
        self.scaler = torch.amp.GradScaler(self.device, enabled=(self.dtype == 'float16')) #if not working in float16, acts as a no-op
        #BCE loss function
        self.criterion = nn.BCEWithLogitsLoss()
        #dataloaders
        self.train_loader = DataLoader(
            OBGAudioDataset(self.train_path, self.sequence_len),
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            pin_memory=self.pin_memory #for faster and async gpu transfers

        )
        self.validation_loader = DataLoader(
            OBGAudioDataset(self.validation_path, self.sequence_len),
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            pin_memory=self.pin_memory #for faster and async gpu transfers
        )
        self.test_loader = DataLoader(
            OBGAudioDataset(self.test_path, self.sequence_len),
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            pin_memory=self.pin_memory #for faster and async gpu transfers
        )
