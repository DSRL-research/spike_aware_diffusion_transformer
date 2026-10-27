import torch
import torch.nn as nn

from configs.base import DetectorConfig

SAME_HOUR_DAYS = 7


class RecurrentBlock(nn.Module):
    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        d_model, d_state = cfg.d_model, cfg.d_state
        kernel = cfg.d_conv - 1 if cfg.d_conv % 2 == 0 else cfg.d_conv
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2, groups=d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.u_to_state = nn.Linear(d_model, d_state)
        self.alpha = cfg.alpha
        self.state_proj = nn.Linear(d_state, d_model)
        self.register_buffer("init_state", torch.zeros(1, d_state))
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        u, v = self.in_proj(x_conv).chunk(2, dim=-1)
        upd = torch.tanh(self.u_to_state(u))
        state = self.init_state.expand(B, -1)
        states = []
        for t in range(T):
            state = self.alpha * state + (1 - self.alpha) * upd[:, t]
            states.append(state)
        h = self.state_proj(torch.tanh(torch.stack(states, dim=1)))
        h = self.dropout(torch.sigmoid(v) * h)
        x = self.norm1(x + h)
        return self.norm2(x + self.ff(x))


class SpikeDetector(nn.Module):
    def __init__(self, cfg: DetectorConfig, horizon: int, history: int, n_hist_feat: int, n_fut_feat: int):
        super().__init__()
        if cfg.head not in ("lead_mlp", "same_hour"):
            raise ValueError(f"Unknown detector head '{cfg.head}'")
        D = cfg.d_model
        self.cfg, self.horizon = cfg, horizon
        self.feat_proj = nn.Linear(1 + n_hist_feat, D)
        self.blocks = nn.ModuleList([RecurrentBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(D)
        self.lead_emb = nn.Parameter(torch.randn(horizon, 16) * 0.1)
        d_in = D + n_fut_feat + 16
        if cfg.head == "same_hour":
            sel = torch.zeros(horizon, SAME_HOUR_DAYS, history)
            for h in range(horizon):
                for j in range(1, SAME_HOUR_DAYS + 1):
                    sel[h, j - 1, history - 24 * j + h] = 1.0
            self.register_buffer("same_hour_select", sel, persistent=False)
            self.same_hour_proj = nn.Sequential(nn.Linear(SAME_HOUR_DAYS * D, D), nn.GELU())
            d_in += D + SAME_HOUR_DAYS
        self.head_mlp = nn.Sequential(nn.Linear(d_in, cfg.head_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                      nn.Linear(cfg.head_hidden, 1))

    def forward(self, hist, hist_feat, fut_feat):
        x = self.feat_proj(torch.cat([hist, hist_feat], dim=-1))
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        last = x[:, -1, :]
        B = x.size(0)
        parts = [last.unsqueeze(1).expand(B, self.horizon, -1), fut_feat, self.lead_emb.unsqueeze(0).expand(B, -1, -1)]
        if self.cfg.head == "same_hour":
            states = torch.einsum("fjl,bld->bfjd", self.same_hour_select, x).flatten(2)
            values = torch.einsum("fjl,bl->bfj", self.same_hour_select, hist[..., 0])
            parts += [self.same_hour_proj(states), values]
        return self.head_mlp(torch.cat(parts, dim=-1))
