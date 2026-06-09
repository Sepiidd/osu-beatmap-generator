import torch
import torch.nn as nn
from dataclasses import dataclass

class OnsetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #TODO: define layers
        self.conv1 = torch.Conv2d(
                    in_channels=self.config.n_in1,
                    out_channels=self.config.nout_1,
                    kernel_size=(self.config.k_len1, self.config.k_wid1)
                )
        self.maxpool1 = nn.MaxPool1d(3, stride=3)

        self.conv2 = torch.Conv2d(
                    in_channels=self.config.nout1,
                    out_channels=self.config.nout2,
                    kernel_size=(self.config.k_len2, self.config.k_wid2)
                )
        self.maxpool2 = nn.MaxPool1(3, stride=3)

        

    def forward(self, x):
        """ 
        TODO: reorder input for convolution layers to be shape (B*S, 3, 15, 80) i.e. (B*S, W, T, F) where B is batch size, S is sequence length, W is number of STFT window lengths, 
        T is the temporal dimension, and F is the mel bin count representing frequency buckets
        """
        #TODO: perform computation
        pass



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

    #attention blocks
    block_size: int = 512 #sequence maximum length
    vocab_size: int = 3012 #tokenizer.py
    n_layer: int = 12 #number of transformer layers (EncoderBlock module)
    n_head: int = 12 #number of attention heads per EncoderBlock
    n_embd: int = 768 #embedding layer size
    #dropout: float = 0.0 #copied over from csc413 example
    #bias: bool = True #copied over from csc413 example

if __name__ == "__main__":
    model = OnsetModel(OnsetConfig())
    print("done")
