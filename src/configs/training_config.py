from dataclasses import dataclass
from dataclass.obg_audio_dataset import OBGAudioDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from pathlib import Path
from configs.audio_config import AudioConfig
from configs.preprocess_config import PreprocessConfig 
from preprocess.audio_utils import augment_speed
from preprocess.audio_utils import augment_pitch
from preprocess.audio_utils import augment_frequency_mask
from preprocess.audio_utils import augment_temporal_mask
import torch
import torch.nn as nn
import torchvision.ops as ops

BASE_DIR = Path(__file__).parent
SEQ_LEN = AudioConfig().sequence_len

configP = PreprocessConfig()

#not dataclass requires more logic
class TrainingConfig():
    def __init__(self):
        #==========STATIC VALUES==========
        #training iteration breakpoints
        self.sequence_len = SEQ_LEN
        self.max_iters = None #set to None if using early stopping via aucpr_goal
        self.log_interval = 10
        self.evaluations = 25 
        self.eval_iters = 50 #factor of checkpoint_iters
        self.warmup_iters = 500 
        self.lr_decay_iters = 3000 #TODO: experiment with this and warmup_iters 
        self.checkpoint_iters = 50 #multiple of eval_iters
        self.aucpr_goal = 0.8 #early stopping threshold
        #model training specifics
        self.weight_decay = 0.1 #adamw
        self.lr = 3e-4
        self.beta1 = 0.9 #adamw
        self.beta2 = 0.99 #adamw
        self.min_lr = 3e-5 #TODO: experiment with this
        self.decay_lr = True #variable learning rate scheduler
        self.grad_clip = 5.0 #TODO: play with this
        self.grad_accumulation_steps = 4
        #dataloaders
        self.train_path = Path(configP.h5_parent) / "datasets" / "train"
        self.validation_path = Path(configP.h5_parent) / "datasets" / "validation" 
        self.test_path = Path(configP.h5_parent) / "datasets" / "test"
        self.checkpoint_path = Path(configP.checkpoint) / "onset_checkpoints"
        self.batch_size = 32
        self.shuffle = True
        self.pin_memory = True
        #criterion
#        self.criterion = nn.BCEWithLogitsLoss() #binary cross entropy
        self.crit_alpha = 0.75
        self.crit_gamma = 2.0
        self.crit_reduction = "mean" 
        self.criterion = lambda logits, targets: ops.sigmoid_focal_loss(logits, targets, alpha=self.crit_alpha, gamma=self.crit_gamma, reduction=self.crit_reduction) #focal loss
        #checkpointing
        self.max_checkpoints = 5

        #==========COMPUTED VALUES==========
        #check if cuda is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #check if gradscaler is required
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        self.pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=self.pt_dtype)
        self.scaler = torch.amp.GradScaler(self.device, enabled=(self.dtype == 'float16')) #if not working in float16, acts as a no-op
        #dataloaders
        a_frq = lambda x, y: (augment_frequency_mask(x), y) #inputs, targets
        a_time = lambda x, y: (augment_temporal_mask(x), y) #inputs, targets
        self.train_loader = DataLoader(
            OBGAudioDataset(self.train_path, self.sequence_len, augment=True, augmentations=[a_frq, a_time]),
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            pin_memory=self.pin_memory #for faster and async gpu transfers
        )

        train_eval_data = OBGAudioDataset(self.train_path, self.sequence_len, full=False)
        self.train_eval = DataLoader(
            train_eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory, #for faster and async gpu transfers
            sampler=RandomSampler(train_eval_data, replacement=True) 
        )
        validation_eval_data = OBGAudioDataset(self.validation_path, self.sequence_len, full=False)
        self.validation_loader = DataLoader(
            validation_eval_data,
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=self.pin_memory, #for faster and async gpu transfers
            sampler=RandomSampler(validation_eval_data, replacement=True)
        )
        test_eval_data = OBGAudioDataset(self.test_path, self.sequence_len, full=False)
        self.test_loader = DataLoader( 
            test_eval_data,
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=self.pin_memory, #for faster and async gpu transfers
            sampler=RandomSampler(test_eval_data, replacement=True)
        )
