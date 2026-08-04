import torch
from torch import Tensor
import torch.nn as nn
import math

class DifficultyEncodingFilm(nn.Module):
    def __init__(self, n_embd: int, n_conditioning: int):
        super().__init__()
        self.film_gen = nn.Linear(n_conditioning, 2*n_embd) #check size

    def forward(self, x: Tensor, difficulty_conditioning: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #TODO
        #apply film_gen to difficulty_conditioning
        #split result into gamma, beta
        gamma, beta = self.film_gen(difficulty_conditioning).split(n_embd, dim=2)
        #point-wise multiply gamma
        x = x * gamma
        #add beta bias
        x = x + beta
        return x
