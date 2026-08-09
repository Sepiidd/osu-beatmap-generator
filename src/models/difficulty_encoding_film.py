import torch
import torch.nn as nn
import math
from torch import Tensor

class DifficultyEncodingFilm(nn.Module):
    def __init__(self, n_embd: int, n_conditioning: int):
        super().__init__()
        self.n_embd = n_embd
        self.film_gen = nn.Linear(n_conditioning, 2*n_embd) #check size

    def forward(self, x: Tensor, difficulty_conditioning: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #apply film_gen to difficulty_conditioning
        #split result into gamma, beta
        gb = self.film_gen(difficulty_conditioning)
        gb = gb.unsqueeze(1)
        gamma, beta = gb.split(self.n_embd, dim=2)
        #point-wise multiply gamma
        x = x * gamma
        #add beta bias
        x = x + beta
        return x
