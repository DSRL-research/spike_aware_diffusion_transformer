import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from configs.base import ExperimentConfig
from src.models import build_detector
from src.train.common import EarlyStopping, cpu_state, resolve_device

KEYS = ("hist", "hist_feat", "fut_feat", "label", "window")


def materialize(ds, device) -> dict:
    items = [ds[i] for i in range(len(ds))]
    return {k: torch.stack([it[k] for it in items]).to(device) for k in KEYS}


class DetectorTrainer:
    def __init__(self, cfg: ExperimentConfig, datasets: dict, seed: int):
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.series = datasets["train"].series
        self.model = build_detector(cfg, self.series).to(self.device)
        self.tensors = {s: materialize(ds, self.device) for s, ds in datasets.items()}
        self.gen = torch.Generator().manual_seed(seed)
        self.prevalence_train = float(self.tensors["train"]["label"][..., 0].cpu().numpy().mean())
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.best_state = None

    def batches(self, split: str, shuffle: bool):
        bs = self.cfg.train.batch_size
        t = self.tensors[split]
        n = len(t["label"])
        order = torch.randperm(n, generator=self.gen) if shuffle else torch.arange(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs].to(self.device)
            yield {k: v[idx] for k, v in t.items()}

    def logits(self, b: dict) -> torch.Tensor:
        return self.model(b["hist"], b["hist_feat"], b["fut_feat"])

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for b in self.batches("train", shuffle=True):
            self.opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self.logits(b), b["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.opt.step()
            total += loss.item()
            n += 1
        return total / max(1, n)

    @torch.no_grad()
    def predict(self, split: str) -> dict:
        self.model.eval()
        logits, labels, windows = [], [], []
        for b in self.batches(split, shuffle=False):
            logits.append(self.logits(b)[..., 0].cpu())
            labels.append(b["label"][..., 0].cpu())
            windows.append(b["window"].cpu())
        return dict(logits=torch.cat(logits).numpy(), labels=torch.cat(labels).numpy(),
                    window=torch.cat(windows).numpy())

    def fit(self) -> list:
        stopper = EarlyStopping(self.cfg.train.patience)
        history = []
        for epoch in range(1, self.cfg.train.epochs + 1):
            start = time.time()
            train_loss = self.train_epoch()
            val = self.predict("val")
            auprc = float(average_precision_score(val["labels"].ravel(), val["logits"].ravel()))
            improved = stopper.step(-auprc)
            if improved:
                self.best_state = cpu_state(self.model)
            history.append(dict(epoch=epoch, train_loss=train_loss, val_auprc=auprc, best=improved,
                                seconds=round(time.time() - start, 1)))
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val AUPRC {auprc:.4f}{'  *' if improved else ''}",
                  flush=True)
            if stopper.stop:
                break
        self.model.load_state_dict(self.best_state)
        return history

    def all_logits(self, windows: int, splits=("train", "val", "test")) -> np.ndarray:
        out = np.full((windows, self.cfg.data.horizon), np.nan, dtype=np.float32)
        for split in splits:
            pred = self.predict(split)
            out[pred["window"]] = pred["logits"].reshape(len(pred["window"]), -1)
        return out
