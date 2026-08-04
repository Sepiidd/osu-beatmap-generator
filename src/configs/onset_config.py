from dataclasses import dataclass
from configs.audio_config import AudioConfig

configA = AudioConfig()

@dataclass
class OnsetConfig:
    #input sizes (for computation)
    time: int = 15
    freq: int = 80
    features: int = 3

    #convolutions
    n_in1: int = 3 
    k_len1: int = 7 
    k_wid1: int = 3 
    n_out1: int = 10 #equal to n_in2
    n_out2: int = 20
    k_len2: int = 3 
    k_wid2: int = 3 

    #pooling
    pool1_width: int = 3 
    pool1_stride: int = 3 
    pool2_width: int = 3
    pool2_stride: int = 3 

    #transformer setup
#    term_t: int = time + (-k_len1+1) + (-k_len2+1) #time size after cnn operations
#    term_f: int = freq + (-k_wid1+1) + (-k_wid2+1) #frequency size after cnn operations
#    term_ft: int = 20 #output features from cnn operations
    conv_out_size: int = 1120 #flattened output size after convolutions and pooling operations (work this math manually)
#    conv_out_size: int = term_t * term_f * term_ft #flattened output size after convolutions and pooling operations 

    #attention blocks
    block_size: int = configA.sequence_len #sequence maximum length
    vocab_size: int = 3012 #tokenizer.py, dont need this lol
    n_layer: int = 12 #number of transformer layers (EncoderBlock module)
    n_head: int = 12 #number of attention heads per EncoderBlock, NOTE: n_embd % n_head = 0
    n_embd: int = 768 #embedding layer size, NOTE: n_embd % n_head = 0
    dropout: float = 0.1
    bias: bool = True #copied over from csc413 example

    #difficulty conditioning (FiLM)
    n_conditioning: int = 3
