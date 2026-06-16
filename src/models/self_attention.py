import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module): 
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #attributes related to attention
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        #key, query, value linear transformations all in one layer
        self.qkv_proj = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)

        #attention
        self.attn_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") #flash attention 

        #output projection
        self.residual_dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x): 
        B, S, D = x.shape #batch, sequence length, dimensionality of embedding

        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(B, S, self.n_head, D // self.n_head).transpose(1, 2) #reorganize according to number of attention heads, reshape to (B, num_head, S, head_size)
        k = k.view(B, S, self.n_head, D // self.n_head).transpose(1, 2) #reorganize according to number of attention heads, reshape to (B, num_head, S, head_size)
        v = v.view(B, S, self.n_head, D // self.n_head).transpose(1, 2) #reorganize according to number of attention heads, reshape to (B, num_head, S, head_size)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            #manual dot product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v #(B, nh, S, S) -> (B, nh, S, hs)
        #fold heads back into eachother
        y = y.transpose(1, 2).contiguous().view(B, S, D) #contiguous is required due to operating on transpositions, memory is not in same order as indexing

        #out projections
        y = self.residual_dropout(self.out_proj(y))
        return y


