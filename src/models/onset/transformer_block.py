import torch
import torch.nn as nn
from models.layer_norm import LayerNorm
from models.self_attention import SelfAttention 

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x): 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module): #final sigmoid layer for attention block
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.n_embd, 3*config.n_embd)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x): 
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

