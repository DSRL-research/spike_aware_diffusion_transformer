import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VestasConfig:
    sequence_length: int = 72
    forecast_horizon: int = 24
    temporal_embedding_dim: int = 12
    condition_dim: int = 17
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 5e-4
    epochs: int = 50


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h; self.dh = d // h; self.sc = (d // h) ** -0.5
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask=None):
        B, T, D = x.shape
        def proj(l): return l(x).view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = proj(self.q), proj(self.k), proj(self.v)
        s = torch.matmul(q, k.transpose(-2, -1)) * self.sc
        if mask is not None: s = s.masked_fill(mask == 0, -1e9)
        a = self.drop(torch.softmax(s, dim=-1))
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(B, T, D))

class ClassicFFN(nn.Module):
    def __init__(self, d, m=4, drop=0.0, act='relu'):
        super().__init__()
        self.fc_in  = nn.Linear(d, m * d)
        self.fc_out = nn.Linear(m * d, d)
        self.drop   = nn.Dropout(drop)
        self.act_fn = F.relu if act == 'relu' else F.gelu

    def forward(self, x):
        return self.fc_out(self.drop(self.act_fn(self.fc_in(x))))

def build_sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.attn = MultiHeadAttention(d, h, drop)
        self.ff   = ClassicFFN(d, 4, drop)
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = self.n1(x + self.attn(x, mask))
        x = self.n2(x + self.ff(x))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        in_dim = 1 + cfg.temporal_embedding_dim
        self.input_proj = nn.Linear(in_dim, cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.head = nn.Linear(cfg.d_model + cfg.forecast_horizon,
                              cfg.forecast_horizon)

        self.register_buffer(
            "pos_emb",
            build_sinusoidal_pe(cfg.sequence_length, cfg.d_model),
            persistent=False,
        )

    def forward(self, hist, temp_emb, fut_cond):
        B, L, _ = hist.shape

        x = torch.cat([hist, temp_emb], dim=-1)
        x = self.input_proj(x) + self.pos_emb[:, :L, :]

        for blk in self.blocks:
            x = blk(x)

        summary = x.mean(dim=1)
        fut_flat = fut_cond.squeeze(-1)
        out = self.head(torch.cat([summary, fut_flat], dim=-1))
        return out.unsqueeze(-1)
