import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.base import ROOT

DLINEAR_PATH = ROOT / "third_party" / "LTSF-Linear" / "models" / "DLinear.py"
TRAINING = dict(learning_rate=0.001, batch_size=16, train_epochs=10, patience=3)


def dlinear_module():
    name = "ltsf_dlinear"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DLINEAR_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def build_model(history: int, horizon: int):
    return dlinear_module().Model(SimpleNamespace(seq_len=history, pred_len=horizon, individual=False, enc_in=1))


class DLinearWindows(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base[i]
        return {"x": it["hist"], "y": it["fut"], "window": it["window"], "mean": it["mean"], "std": it["std"],
                "fut_raw": it["fut_raw"], "label": it["label"][:, 0]}


def adjust_learning_rate(optimizer, epoch: int, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * (0.5 ** ((epoch - 1) // 1))


@torch.no_grad()
def predict(model, loader, device) -> dict:
    model.eval()
    out = {k: [] for k in ("pred_norm", "y_true", "spike", "mean", "std", "window")}
    for b in loader:
        out["pred_norm"].append(model(b["x"].to(device))[:, :, 0].cpu().numpy())
        out["y_true"].append(b["fut_raw"].numpy())
        out["spike"].append(b["label"].bool().numpy())
        for k in ("mean", "std", "window"):
            out[k].append(b[k].numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["hist_end"] = loader.dataset.base.series.windows["hist_end"].to_numpy()[res["window"]]
    return res
