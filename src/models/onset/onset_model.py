import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.position_encoding import PositionalEncoding
from models.layer_norm import LayerNorm
from models.onset.transformer_block import EncoderBlock
from configs.onset_config import OnsetConfig

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
        self.maxpool1 = nn.MaxPool1d(config.pool1_width, stride=config.pool1_stride)

        self.conv2 = nn.Conv2d(
                    in_channels=config.n_out1,
                    out_channels=config.n_out2,
                    kernel_size=(config.k_len2, config.k_wid2)
                )
        self.maxpool2 = nn.MaxPool1d(config.pool2_width, stride=config.pool2_stride)

        #project to transformer embedding size
        self.lin1 = nn.Linear(config.conv_out_size, config.n_embd)

        #multi-head attention, encoder only
        self.encoder = nn.ModuleDict(dict(
                pe = PositionalEncoding(config.n_embd, max_len=config.block_size),
                drop = nn.Dropout(config.dropout),
                ah = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
                ln = LayerNorm(config.n_embd, bias=config.bias)
            ))

        #final linear layers
#        self.lin2 = nn.Linear(config.n_embd, 1, bias=False)
        self.lin2 = nn.Linear(config.n_embd, config.n_embd // 2, bias=False)
        self.lin3 = nn.Linear(config.n_embd // 2, 1, bias=False)

        #initial weight randomization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj"):
                #gpt2 scaled init for residual projections (for attention)
                torch.nn.init._normal(p, mean=0.0, std=0.02/math.sqrt(2 * config)) #2 comes from amount of residual connections performed


    def forward(self, x):
        """ 
        input <x> is expected to come in with shape (B, S, 15, 80, 3)
        """
        #reorder to shape (B*S, W, T, F)
        B, S, T, F, W = x.shape #batch size, seq len, stft time, mel freq, stft window size
        x = x.view(B*S, T, F, W) #fold seq len into batch size for convolutions
        x = x.permute(0, 3, 1, 2) #reorder as required by convolutions

        #convolutions
        x = self.conv1(x)
        #reshape, fold
        BS, C, T, F = x.shape #batch size * seq len, output channels, stft time, mel freq
        x = x.permute(0, 2, 1, 3) #move stft time dimension next to batch/seq_len dimension
        x = x.reshape(BS * T, C, F)
        x = self.maxpool1(x)
        #unfold, reshape
        x = x.reshape(BS, T, C, -1)
        x = x.permute(0, 2, 1, 3)

        x = self.conv2(x)
        #reshape, fold
        BS, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(BS * T, C, F)
        x = self.maxpool2(x)
        #unfold, reshape
        x = x.reshape(BS, T, C, -1)
        x = x.permute(0, 2, 1, 3)

        #unroll batch, sequence dimension
        x = x.reshape(B, S, -1)

        #project size, encode info
        x = self.lin1(x)
        x = self.encoder.pe(x)

        #attention blocks
        x = self.encoder.drop(x)
        for block in self.encoder.ah:
            x = block(x)
        x = self.encoder.ln(x)
        
        #obtain logits from attended info
        x = self.lin2(x)
        logits = self.lin3(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        #get all learnable parameters
        p_dct = {pn: p for pn, p in self.named_parameters()}
        p_dct = {pn: p for pn, p in p_dct.items() if p.requires_grad}
        
        #optimizer groups
        decay_params = [p for n,p in p_dct.items() if p.dim()>=2]
        nodecay_params = [p for n,p in p_dct.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        #element number and size tracking
        num_decay = sum(p.numel() for p in decay_params)
        size_decay = sum(p.nbytes for p in decay_params)
        num_nodecay = sum(p.nbytes for p in nodecay_params)
        size_nodecay = sum(p.nbytes for p in nodecay_params)
        print(f"onset model has {len(decay_params)} many decay tensors, with a total of {num_decay} parameters taking up {size_decay} many bytes")
        print(f"onset model has {len(nodecay_params)} many nodecay tensors, with a total of {num_nodecay} parameters taking up {size_nodecay} many bytes")

        #check if fused implementation of adamw is available
        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        #create AdamW
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

if __name__ == "__main__":
    model = OnsetModel(OnsetConfig())
    print("done")
