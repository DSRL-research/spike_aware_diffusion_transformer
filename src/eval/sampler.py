import torch

from configs.base import DiffusionConfig


@torch.no_grad()
def ddim_sample(model, noise, batch: dict, diff_cfg: DiffusionConfig, horizon: int, p_t=None, generator=None):
    hist, hist_feat, fut_feat = batch["hist"], batch["hist_feat"], batch["fut_feat"]
    B = hist.size(0)
    grid = torch.linspace(0, diff_cfg.timesteps - 1, diff_cfg.ddim_steps, dtype=torch.long)
    x = noise.sample((B, horizon, 1), batch["context"], batch["hour"], p_t=p_t, generator=generator)
    w = diff_cfg.x0_spike_scale
    for i in reversed(range(1, diff_cfg.ddim_steps)):
        t_cur, t_prev = int(grid[i]), int(grid[i - 1])
        t = torch.full((B,), t_cur, device=hist.device, dtype=torch.long)
        eps, x0 = model(x, hist, t, hist_feat, fut_feat)
        if w != 0.0:
            x0 = x0 * (1.0 + p_t * w)
        x = noise.sqrt_ac[t_prev] * x0 + torch.sqrt(1 - noise.ac[t_prev]) * eps
    return x
