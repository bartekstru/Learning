import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class CausalSelfAttention(nn.Module):
    """
    Vanilla multi head masked self-attention with a projection at the end.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query and value (3) projections for all heads (n_head, each of size head_size), but in a batch
        # n_head * head_size = n_embed
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # each projection is (n_embed, head_size), and we have 3 * n_heads, so the "batch" size is (n_embed, head_size * n_heads * 3) = (n_embed, n_embed)

        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed) # (head_size*num_heads, n_embed) = (n_embed, n_embed)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop) # dropout to apply after the attention is calculated
        self.residual_dropout = nn.Dropout(config.resid_pdrop) # dropout to apply after the residual connection

        # causal mask to ensure that attentions is applied only to the characters from the past (to the left in the input sequence)
        self.register_buffer("bias", torch.
                             tril(torch.ones(config.block_size, config.block_size)).    # attend only to previous positions in a sequence
                             view(1, 1, config.block_size, config.block_size)           # add dimensions for batch size and the number of heads in the attention mechanism -> (1, 1, block_size, block_size)
                            )
        
        self.n_embed = config.n_embed
        self.n_head = config.n_head

    def forward(self, x : torch.Tensor):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embed, 2) # divide (batch_size, block_size, 3*n_embed) into 3 chunks along the last dimension
        # C // self.n_head is equal to  head_size (hence the assertion in the init function)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (batch_size, n_heads, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (batch_size, n_heads, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (batch_size, n_heads, T, head_size)

        # causal self-attention - (batch_size, n_heads, T, head_size) @ (batch_size, n_heads, head_size, T)
        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(C // self.n_head))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att) # (batch_size, n_heads, T, T)
        y = att @ v # (batch_size, n_heads, T, head_size)
        y = y.transpose(1,2).contiguous().view(B,T,C) # (batch_size, block_size, n_embed)
        return self.residual_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(config.n_embed, 4 * config.n_embed),
            c_proj = nn.Linear(config.n_embed * 4, config.n_embed),
            act = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop)
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # forward pass of MLP

    def forward(self, x):
        x = x + self.attn(self.ln_1)
        x = x + self.mlpf(self.ln_2)
        return x

