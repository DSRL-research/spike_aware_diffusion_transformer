import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ModelConfig
from src.models.layers import MultiHeadAttention, SinusoidalEmbedding, causal_mask, lag_index


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, num_steps: int, dim: int):
        super().__init__()
        half = dim // 2
        steps = np.arange(num_steps)[:, None]
        freqs = 10.0 ** (np.arange(half) / (half - 1) * 4.0)[None, :]
        table = np.concatenate([np.sin(steps * freqs), np.cos(steps * freqs)], axis=1)
        self.register_buffer("diffusion_embedding", torch.tensor(table, dtype=torch.float32), persistent=False)
        self.projection1 = nn.Linear(dim, dim)
        self.projection2 = nn.Linear(dim, dim)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        x = self.diffusion_embedding[step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        inner = mult * d_model
        self.fc_in = nn.Linear(d_model, 2 * inner)
        self.fc_out = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        v, g = self.fc_in(x).chunk(2, dim=-1)
        return self.fc_out(self.dropout(v * F.silu(g)))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))


class DiffusionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, history: int, horizon: int, timesteps: int, n_hist_feat: int, n_fut_feat: int):
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        if cfg.x0_residual not in ("none", "same_hour_mean", "last"):
            raise ValueError(f"Unknown x0_residual '{cfg.x0_residual}'")
        self.time_emb = SinusoidalEmbedding(D)
        self.diff_emb = DiffusionStepEmbedding(timesteps, D)
        self.diffusion_projection = nn.Linear(D, D)
        self.feat_enc = nn.Linear(n_hist_feat, D)
        if cfg.future_calendar:
            self.fut_cal_proj = nn.Linear(n_fut_feat, D)
        if cfg.future_lags or cfg.x0_residual == "same_hour_mean":
            self.register_buffer("lag_index", lag_index(history, horizon), persistent=False)
        if cfg.future_lags:
            self.fut_lag_proj = nn.Linear(8, D)
        self.hist_proj = nn.Linear(1, D // 4)
        self.future_proj = nn.Linear(1, D // 4)
        self.inp_fusion = nn.Linear(D + D // 4, D)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(D)
        self.pred_eps = nn.Linear(D, 1)
        self.pred_x0 = nn.Linear(D, 1)
        self.diff_every = cfg.diff_inject_every
        self.diff_gates = nn.Parameter(torch.zeros(math.ceil(cfg.n_layers / self.diff_every)))
        self.pos_emb = nn.Parameter(torch.randn(1, history + horizon, D))

    def forward(self, x_t, hist, t, hist_feat, fut_feat):
        B, Fh, _ = x_t.shape
        L = hist.shape[1]
        ctx = self.feat_enc(hist_feat) + self.time_emb(t).unsqueeze(1)
        hist_tok = self.inp_fusion(torch.cat([ctx, self.hist_proj(hist)], dim=-1))
        fut_ctx = ctx[:, -1:, :].expand(B, Fh, -1)
        if self.cfg.future_calendar:
            fut_ctx = fut_ctx + self.fut_cal_proj(fut_feat)
        if self.cfg.future_lags:
            lags = hist[:, self.lag_index, 0]
            fut_ctx = fut_ctx + self.fut_lag_proj(torch.cat([lags, lags.mean(-1, keepdim=True)], dim=-1))
        fut_tok = self.inp_fusion(torch.cat([fut_ctx, self.future_proj(x_t)], dim=-1))
        seq = torch.cat([hist_tok, fut_tok], dim=1) + self.pos_emb[:, :L + Fh, :]
        mask = causal_mask(L + Fh, seq.device)
        diff = self.diffusion_projection(self.diff_emb(t)).unsqueeze(1)
        for i, blk in enumerate(self.blocks):
            if i % self.diff_every == 0:
                g = torch.sigmoid(self.diff_gates[i // self.diff_every])
                seq = torch.cat([seq[:, :L], seq[:, L:] + g * diff], dim=1)
            seq = blk(seq, mask)
        fut = self.final_norm(seq)[:, L:, :]
        x0 = self.pred_x0(fut)
        if self.cfg.x0_residual == "same_hour_mean":
            x0 = x0 + hist[:, self.lag_index, 0].mean(-1, keepdim=True)
        elif self.cfg.x0_residual == "last":
            x0 = x0 + hist[:, -1:, :]
        return self.pred_eps(fut), x0
