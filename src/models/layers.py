import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
        angles = t[:, None] * freq[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads
        self.scale = self.h_dim ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.h_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


def causal_mask(T: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)


def lag_index(history: int, horizon: int, days: int = 7) -> torch.Tensor:
    if horizon > 24 or history < 24 * days:
        raise ValueError("same-hour lags need horizon <= 24 and history >= 168")
    return torch.tensor([[history - 24 * d + j for d in range(1, days + 1)] for j in range(horizon)])
