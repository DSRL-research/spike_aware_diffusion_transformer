import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import ExperimentConfig
from src.datasets import make_loader, to_device
from src.models import build_transformer
from src.train.common import EMA, EarlyStopping, cpu_state, resolve_device


class TransformerTrainer:
    def __init__(self, cfg: ExperimentConfig, datasets: dict, seed: int):
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.series = datasets["train"].series
        self.model = build_transformer(cfg, self.series).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.ema = EMA(self.model, cfg.train.ema_decay)
        t = cfg.train
        self.train_loader = make_loader(datasets["train"], t.batch_size, shuffle=True, seed=seed, drop_last=True,
                                        num_workers=t.num_workers)
        self.val_loader = make_loader(datasets["val"], t.batch_size, shuffle=True, seed=seed + 1,
                                      num_workers=t.num_workers)
        self.best_state = None

    def loss(self, batch: dict) -> torch.Tensor:
        b = to_device(batch, self.device)
        pred = self.model(b["hist"], b["hist_feat"], b["fut_feat"], future=b["fut"])
        return F.mse_loss(pred, b["fut"])

    def _run(self, loader, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        for batch in loader:
            if train:
                self.opt.zero_grad(set_to_none=True)
                loss = self.loss(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.opt.step()
                self.ema.update(self.model)
            else:
                with torch.no_grad():
                    loss = self.loss(batch)
            total += loss.item()
            n += 1
        return total / max(1, n)

    def fit(self) -> list:
        stopper = EarlyStopping(self.cfg.train.patience)
        history = []
        for epoch in range(1, self.cfg.train.epochs + 1):
            start = time.time()
            train_loss = self._run(self.train_loader, train=True)
            val_loss = self._run(self.val_loader, train=False)
            improved = stopper.step(val_loss)
            if improved:
                self.best_state = cpu_state(self.ema.model)
            history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, best=improved,
                                seconds=round(time.time() - start, 1)))
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}{'  *' if improved else ''}",
                  flush=True)
            if stopper.stop:
                break
        return history

    def eval_model(self):
        model = build_transformer(self.cfg, self.series)
        model.load_state_dict(self.best_state)
        return model.to(self.device).eval()
