import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #TODO: define layers

    def forward(self, x): 
        #TODO: perform computation
        pass

class SelfAttention(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        #TODO: define layers

    def forward(self, x): 
        #TODO: perform computation
        pass

class MLP(nn.Module): #final sigmoid layer for attention block
    def __init__(self, config):
        super().__init__()
        self.config = config
        #TODO: define layers

    def forward(self, x): 
        #TODO: perform computation
        pass
