from dataclasses import dataclass, field
from tokens.tokenizer import obg_tokenizer
from preprocess.converter import obj_converter

@dataclass
class GenConfig:
    prediction_threshold: float = 0.3
    hamming_window_len: int = 9 #with sr=22050, approx 0.4 ms, NOTE: MUST BE ODD TO MATCH LENGTH OF HAMMING WINDOW OUTPUT TO INPUT SEQUENCE LENGTH
    overlap_len: int = 1024 
    batch_size = 8
