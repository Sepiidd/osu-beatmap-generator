import torch
from torch import Tensor
import torch.nn as nn
import math

#class copied over from https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # shape (max_len, 1), looks like [0, ..., max_len-1]
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd)) #list of value to divide position by (represented as 1/{10000^{2i/n_embd}})
        pe = torch.zeros(max_len, 1, n_embd)   
        pe[:, 0, 0::2] = torch.sin(position * div_term) #apply sin pos-encoding on every even term of the n_embd dimension
        pe[:, 0, 1::2] = torch.cos(position * div_term) #apply cos pos-encoding on every odd term of the n_embd dimension
        self.register_buffer('pe', pe) #store encoding as part of module state (moves to gpu and other stuff, marked as not a trainable param)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
