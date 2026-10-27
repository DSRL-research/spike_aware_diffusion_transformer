import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ModelConfig
from src.models.layers import MultiHeadAttention, causal_mask, lag_index


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class ReluFFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        self.fc_in = nn.Linear(d_model, mult * d_model)
        self.fc_out = nn.Linear(mult * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_out(self.dropout(F.relu(self.fc_in(x))))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = ReluFFN(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))


class TransformerDecoderAR(nn.Module):
    def __init__(self, cfg: ModelConfig, history: int, horizon: int, n_hist_feat: int, n_fut_feat: int):
        super().__init__()
        D = cfg.d_model
        if cfg.x0_residual not in ("none", "same_hour_mean"):
            raise ValueError(f"Unknown x0_residual '{cfg.x0_residual}'")
        self.cfg, self.horizon = cfg, horizon
        self.feat_enc = nn.Linear(n_hist_feat, D)
        if cfg.future_calendar:
            self.fut_cal_proj = nn.Linear(n_fut_feat, D)
        if cfg.future_lags or cfg.x0_residual == "same_hour_mean":
            self.register_buffer("lag_index", lag_index(history, horizon), persistent=False)
        if cfg.future_lags:
            self.fut_lag_proj = nn.Linear(8, D)
        self.val_proj = nn.Linear(1, D // 4)
        self.inp_fusion = nn.Linear(D + D // 4, D)
        self.register_buffer("pos_emb", sinusoidal_positions(history + horizon, D), persistent=False)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def _next_step(self, values, covariates):
        S = values.shape[1]
        tok = self.inp_fusion(torch.cat([covariates, self.val_proj(values)], dim=-1))
        tok = tok + self.pos_emb[:, :S, :]
        mask = causal_mask(S, values.device)
        for blk in self.blocks:
            tok = blk(tok, mask)
        return self.head(self.final_norm(tok))

    def forward(self, hist, hist_feat, fut_feat, future=None):
        B, L, _ = hist.shape
        H = self.horizon
        ctx_hist = self.feat_enc(hist_feat)
        fut_ctx = ctx_hist[:, -1:, :].expand(B, H, -1)
        extra = torch.zeros_like(fut_ctx)
        if self.cfg.future_calendar:
            extra = extra + self.fut_cal_proj(fut_feat)
        if self.cfg.future_lags:
            lags = hist[:, self.lag_index, 0]
            extra = extra + self.fut_lag_proj(torch.cat([lags, lags.mean(-1, keepdim=True)], dim=-1))
        base = torch.zeros(B, H, 1, device=hist.device, dtype=hist.dtype)
        if self.cfg.x0_residual == "same_hour_mean":
            base = hist[:, self.lag_index, 0].mean(-1, keepdim=True)
        fut_ctx = fut_ctx + extra
        if future is not None:
            values = torch.cat([hist, future[:, :-1, :]], dim=1)
            covariates = torch.cat([ctx_hist, fut_ctx[:, :H - 1]], dim=1)
            return self._next_step(values, covariates)[:, L - 1:, :] + base
        values, covariates, out = hist, ctx_hist, []
        for j in range(H):
            nxt = self._next_step(values, covariates)[:, -1:, :] + base[:, j:j + 1]
            out.append(nxt)
            values = torch.cat([values, nxt], dim=1)
            if j + 1 < H:
                covariates = torch.cat([covariates, fut_ctx[:, j:j + 1]], dim=1)
        return torch.cat(out, dim=1)
