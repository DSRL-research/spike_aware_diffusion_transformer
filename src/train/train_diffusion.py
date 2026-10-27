import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ExperimentConfig
from src.datasets import WindowDataset, make_loader, to_device
from src.models import build_diffusion
from src.train.common import EMA, EarlyStopping, cpu_state, resolve_device


class DiffusionTrainer:
    def __init__(self, cfg: ExperimentConfig, datasets: dict, seed: int, probs=None):
        if cfg.needs_spike_probs and probs is None:
            raise ValueError(f"{cfg.name} needs the spike probabilities of the detector")
        self.cfg, self.seed, self.probs = cfg, seed, probs
        self.device = resolve_device(cfg.train.device)
        self.series = datasets["train"].series
        self.model, self.noise = build_diffusion(cfg, self.series)
        self.model.to(self.device)
        self.noise.to(self.device)
        self._compiled = None
        params = list(self.model.parameters()) + list(self.noise.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.ema = EMA(self.model, cfg.train.ema_decay)
        t = cfg.train
        self.train_loader = make_loader(datasets["train"], t.batch_size, shuffle=True, seed=seed, drop_last=True,
                                        num_workers=t.num_workers)
        self.val_loader = make_loader(datasets["val"], t.batch_size, shuffle=True, seed=seed + 1,
                                      num_workers=t.num_workers)
        if t.select_on not in ("val_loss", "sampled_mae"):
            raise ValueError(f"Unknown select_on '{t.select_on}'")
        self.select_set = None
        if t.select_on == "sampled_mae":
            val = datasets["val"]
            n = t.select_windows or len(val)
            pick = np.unique(np.linspace(0, len(val) - 1, min(n, len(val))).round().astype(int))
            self.select_set = WindowDataset(val.series, val.indices[pick], val.cfg)
        self.best_states = None
        self.nonfinite = 0

    @property
    def train_model(self):
        if not self.cfg.train.compile:
            return self.model
        if self._compiled is None:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._compiled = torch.compile(self.model)
        return self._compiled

    def loss(self, batch: dict) -> torch.Tensor:
        b = to_device(batch, self.device)
        x0 = b["fut"]
        B = b["hist"].size(0)
        t = torch.randint(0, self.cfg.diffusion.timesteps, (B,), device=self.device)
        p_t = self.probs(b) if self.probs is not None else None
        eps = self.noise.sample(x0.shape, b["context"], b["hour"], p_t=p_t)
        x_t = self.noise.add_noise(x0, eps, t)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.train.amp):
            eps_hat, x0_hat = self.train_model(x_t, b["hist"], t, b["hist_feat"], b["fut_feat"])
        eps_hat, x0_hat = eps_hat.float(), x0_hat.float()
        power = self.cfg.train.window_weight_power
        if power:
            w = b["std"].double() ** power
            w = (w / w.mean()).float()[:, None, None]
            loss_eps = (w * (eps_hat - eps) ** 2).mean()
            loss_x0 = (w * (x0_hat - x0) ** 2).mean()
        else:
            loss_eps = F.mse_loss(eps_hat, eps)
            loss_x0 = F.mse_loss(x0_hat, x0)
        lam = self.cfg.diffusion.lambda_eps
        return lam * loss_eps + (1 - lam) * loss_x0

    def _set_mode(self, train: bool) -> None:
        self.model.train(train)
        self.noise.train(train)

    def train_epoch(self) -> float:
        self._set_mode(True)
        clip = self.cfg.train.grad_clip
        total, n = 0.0, 0
        for batch in self.train_loader:
            self.opt.zero_grad(set_to_none=True)
            loss = self.loss(batch)
            if not torch.isfinite(loss):
                self.nonfinite += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            nn.utils.clip_grad_norm_(self.noise.parameters(), clip)
            self.opt.step()
            self.ema.update(self.model)
            total += loss.item()
            n += 1
        return total / max(1, n)

    @torch.no_grad()
    def val_loss(self) -> float:
        self._set_mode(False)
        total, n = 0.0, 0
        for batch in self.val_loader:
            loss = self.loss(batch)
            if not torch.isfinite(loss):
                self.nonfinite += 1
                continue
            total += loss.item()
            n += 1
        return total / max(1, n)

    @torch.no_grad()
    def sampled_mae(self) -> float:
        from src.eval.evaluate import predict
        pred = predict(self.cfg, self.select_set, self.ema.model, self.noise, self.probs, seed=self.seed)
        point = pred["pred_norm"] * pred["std"][:, None] + pred["mean"][:, None]
        return float(np.mean(np.abs(point - pred["y_true"])))

    def fit(self) -> list:
        stopper = EarlyStopping(self.cfg.train.patience)
        history = []
        for epoch in range(1, self.cfg.train.epochs + 1):
            start = time.time()
            self.nonfinite = 0
            train_loss = self.train_epoch()
            val_loss = self.val_loss()
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            if self.select_set is None:
                improved = stopper.step(val_loss)
            else:
                row["val_sampled_mae"] = self.sampled_mae()
                improved = stopper.step(row["val_sampled_mae"])
            if improved:
                self.best_states = dict(model=cpu_state(self.ema.model), noise=cpu_state(self.noise))
            row.update(best=improved, seconds=round(time.time() - start, 1), nonfinite_batches=self.nonfinite)
            history.append(row)
            extra = f"  val MAE {row['val_sampled_mae']:.4f}" if "val_sampled_mae" in row else ""
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}{extra}{'  *' if improved else ''}",
                  flush=True)
            if stopper.stop:
                break
        return history

    def eval_models(self) -> tuple:
        model, noise = build_diffusion(self.cfg, self.series)
        model.load_state_dict(self.best_states["model"])
        noise.load_state_dict(self.best_states["noise"])
        return model.to(self.device).eval(), noise.to(self.device).eval()
