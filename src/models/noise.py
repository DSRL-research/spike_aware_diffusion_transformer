import torch
import torch.nn as nn

from configs.base import DiffusionConfig, NoiseConfig

FAMILIES = {
    "johnson_su": lambda z, a, b: torch.sinh((z - a) / b),
    "gaussian": lambda z, a, b: z,
    "student_t": lambda z, a, b: z,
}


def _draw(fn, shape, device, generator):
    if isinstance(generator, (list, tuple)):
        return torch.cat([fn((1, *shape[1:]), device=device, generator=g) for g in generator])
    return fn(shape, device=device, generator=generator)


class NoiseSampler(nn.Module):
    def __init__(self, noise_cfg: NoiseConfig, diff_cfg: DiffusionConfig, context_dim: int):
        super().__init__()
        if noise_cfg.family not in FAMILIES:
            raise ValueError(f"Unknown noise family '{noise_cfg.family}'")
        if noise_cfg.mixture not in ("none", "laplace"):
            raise ValueError(f"Unknown mixture '{noise_cfg.mixture}'")
        self.cfg = noise_cfg
        self.transform = FAMILIES[noise_cfg.family]
        betas = torch.linspace(diff_cfg.beta_start, diff_cfg.beta_end, diff_cfg.timesteps)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("ac", ac)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_om", torch.sqrt(1 - ac))
        self.register_buffer("hourly_uncertainty_factors", torch.tensor(noise_cfg.hourly_factors))
        layers, d_in, hidden = [], context_dim + 2, tuple(noise_cfg.refiner_hidden)
        for i, h in enumerate(hidden):
            layers += [nn.Linear(d_in, h), nn.LayerNorm(h), nn.ReLU()]
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(noise_cfg.refiner_dropout))
            d_in = h
        layers.append(nn.Linear(d_in, 4))
        self.johnson_refiner = nn.Sequential(*layers)
        nn.init.zeros_(self.johnson_refiner[-1].weight)
        nn.init.zeros_(self.johnson_refiner[-1].bias)
        if noise_cfg.spike_modulation:
            w_loc, w_scale, w_a, w_b = noise_cfg.spike_weights_init
            self.w_spike_loc = nn.Parameter(torch.tensor(w_loc))
            self.w_spike_scale = nn.Parameter(torch.tensor(w_scale))
            self.w_spike_a = nn.Parameter(torch.tensor(w_a))
            self.w_spike_b = nn.Parameter(torch.tensor(w_b))

    def params(self, context: torch.Tensor, hours: torch.Tensor) -> tuple:
        B, device = context.size(0), context.device
        base, c = self.cfg.base, self.cfg
        a = torch.full((B,), base["a"], device=device)
        b = torch.full((B,), base["b"], device=device)
        loc = torch.full((B,), base["loc"], device=device)
        scale = torch.full((B,), base["scale"], device=device)
        hours = hours.long() % 24
        scale = scale * self.hourly_uncertainty_factors[hours]
        hour_norm = (hours.float() % 24) / 24.0
        horizon_norm = torch.ones(B, device=device)
        adj = self.johnson_refiner(torch.cat([context, hour_norm[:, None], horizon_norm[:, None]], dim=-1))
        a = torch.clamp(a + adj[:, 0], *c.clamp_a)
        b = torch.clamp(b + adj[:, 1], *c.clamp_b)
        loc = torch.clamp(loc + adj[:, 2], *c.clamp_loc)
        scale = torch.clamp(scale + adj[:, 3], *c.clamp_scale)
        return a, b, loc, scale

    def step_params(self, shape, context, hours, p_t=None) -> tuple:
        a, b, loc, scale = (x[:, None, None].expand(shape) for x in self.params(context, hours))
        if self.cfg.spike_modulation:
            c = self.cfg
            loc = torch.clamp(loc + self.w_spike_loc * p_t, *c.clamp_loc)
            scale = torch.clamp(scale + self.w_spike_scale * p_t, *c.clamp_scale)
            a = torch.clamp(a + self.w_spike_a * p_t, *c.clamp_a)
            b = torch.clamp(b + self.w_spike_b * p_t, *c.clamp_b)
        return a, b, loc, scale

    def sample(self, shape, context, hours, p_t=None, generator=None):
        if (self.cfg.spike_modulation or self.cfg.mixture != "none") and p_t is None:
            raise ValueError("This noise configuration needs the spike probabilities p_t")
        device = context.device
        a, b, loc, scale = self.step_params(shape, context, hours, p_t)
        z = _draw(torch.randn, shape, device, generator)
        if self.cfg.family == "student_t":
            df = int(self.cfg.student_df)
            g = _draw(torch.randn, (*shape, df), device, generator)
            z = z / torch.sqrt((g ** 2).sum(-1) / df)
        eps = loc + scale * self.transform(z, a, b)
        eps = eps - eps.mean(dim=(1, 2), keepdim=True)
        if self.cfg.mixture == "laplace":
            u = _draw(torch.rand, shape, device, generator) - 0.5
            log_term = torch.log1p(torch.clamp(-2 * u.abs(), min=-1 + 2 ** -24))
            laplace = -self.cfg.laplace_scale * torch.sign(u) * log_term
            eps = (1 - p_t) * eps + p_t * laplace
        return eps

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sqrt_ac[t].view(-1, 1, 1) * clean + self.sqrt_om[t].view(-1, 1, 1) * noise
