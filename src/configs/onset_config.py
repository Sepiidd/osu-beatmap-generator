from dataclasses import dataclass

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
    pool1_width = 3 
    pool1_stride = 3 
    pool2_width = 3
    pool2_stride = 3 

    #transformer setup
    conv_out_size = 1120 #flattened output size after convolutions and pooling operations (work this math manually) n_out2*()*()

    #attention blocks
    block_size: int = 512 #sequence maximum length
    vocab_size: int = 3012 #tokenizer.py, dont need this lol
    n_layer: int = 12 #number of transformer layers (EncoderBlock module)
    n_head: int = 12 #number of attention heads per EncoderBlock, NOTE: n_embd % n_head = 0
    n_embd: int = 384 #embedding layer size, NOTE: n_embd % n_head = 0
    dropout: float = 0.4 
    bias: bool = True #copied over from csc413 example

