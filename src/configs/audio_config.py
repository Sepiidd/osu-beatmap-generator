from dataclasses import dataclass, field
from tokens.tokenizer import obg_tokenizer
from preprocess.converter import obj_converter

@dataclass
class AudioConfig:
    n_fft: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    hop_len: int = 221
    n_mel: int = 80
    sr: int = 22050
    sequence_len: int = 512
