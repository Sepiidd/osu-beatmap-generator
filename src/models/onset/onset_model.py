import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.position_encoding import PositionalEncoding
from models.layer_norm import LayerNorm
from models.onset.transformer_block import EncoderBlock

class OnsetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        #convolutions
        self.conv1 = nn.Conv2d(
                    in_channels=config.n_in1,
                    out_channels=config.n_out1,
                    kernel_size=(config.k_len1, config.k_wid1)
                )
        self.maxpool1 = nn.MaxPool1d(config.pool_width, stride=config.pool_stride)

        self.conv2 = nn.Conv2d(
                    in_channels=config.n_out1,
                    out_channels=config.n_out2,
                    kernel_size=(config.k_len2, config.k_wid2)
                )
        self.maxpool2 = nn.MaxPool1d(config.pool_width, stride=config.pool_stride)

        #project to transformer embedding size
        self.lin1 = nn.Linear(config.conv_out_size, config.n_embd)

        #multi-head attention, encoder only
        self.encoder = nn.ModuleDict(dict(
                pe = PositionalEncoding(config.n_embd, max_len=config.block_size),
                drop = nn.Dropout(config.dropout),
                ah = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
                ln = LayerNorm(config.n_embd, bias=config.bias)
            ))

        #final linear layer
        self.lin2 = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x):
        """ 
        input <x> comes in with shape (B, S, 15, 80, 3)
        """
        #reorder to shape (B*S, W, T, F)
        B, S, T, F, W = x.shape #batch size, seq len, stft time, mel freq, stft window size
        x = x.view(B*S, T, F, W) #fold seq len into batch size for convolutions
        x = x.permute(0, 3, 1, 2) #reorder as required by convolutions

        #convolutions
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        #unroll batch, sequence dimension
        x = x.view(B, S, -1)

        #project size, encode info
        x = self.lin1(x)
        x = self.encoder.pe(x)

        #attention blocks
        x = self.encoder.drop(x)
        for block in self.encoder.ah:
            x = block(x)
        x = self.encoder.ln(x)
        
        #obtain logits from attended info
        logits = self.lin2(x)
        return logits

@dataclass
class OnsetConfig:
    #convolutions
    n_in1: int = 3
    k_len1: int = 7
    k_wid1: int = 3
    n_out1: int = 10 #equal to n_in2
    n_out2: int = 20
    k_len2: int = 3
    k_wid2: int = 3

    #pooling
    pool_width = 3
    pool_stride = 3 

    #transformer setup
    conv_out_size = 1120 #flattened output size after convolutions and pooling operations

    #attention blocks
    block_size: int = 512 #sequence maximum length
    vocab_size: int = 3012 #tokenizer.py, dont need this lol
    n_layer: int = 12 #number of transformer layers (EncoderBlock module)
    n_head: int = 12 #number of attention heads per EncoderBlock
    n_embd: int = 768 #embedding layer size
    dropout: float = 0.0 #copied over from csc413 example
    bias: bool = True #copied over from csc413 example

if __name__ == "__main__":
    model = OnsetModel(OnsetConfig())
    print("done")
