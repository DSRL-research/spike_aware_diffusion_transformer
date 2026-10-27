import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import SeriesData, make_datasets, make_loader
from src.models.vanilla_transformer import VanillaTransformer, VestasConfig
from src.seed import seed_everything

DEVICE = "cuda"


def _inputs(b: dict, tp_col: int):
    return b["hist"].to(DEVICE), b["hist_feat"].to(DEVICE), b["fut_feat"][:, :, tp_col:tp_col + 1].to(DEVICE)


@torch.no_grad()
def predict_vanilla(model, ds, tp_col: int, batch_size: int) -> dict:
    model.eval()
    out = {k: [] for k in ("pred_norm", "y_true", "spike", "mean", "std", "window")}
    for b in make_loader(ds, batch_size, False, 0):
        out["pred_norm"].append(model(*_inputs(b, tp_col))[..., 0].cpu().numpy())
        out["y_true"].append(b["fut_raw"].numpy())
        out["spike"].append(b["label"][..., 0].bool().numpy())
        for k in ("mean", "std", "window"):
            out[k].append(b[k].numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["hist_end"] = ds.series.windows["hist_end"].to_numpy()[res["window"]]
    return res


def run_vanilla(data_cfg, training: dict, seed: int, out: Path) -> tuple:
    seed_everything(seed)
    series = SeriesData(data_cfg)
    ds = make_datasets(data_cfg, series)
    tp_col = series.fut_feature_names.index("future__obs__theoretical_power")
    model = VanillaTransformer(VestasConfig()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    bs = training["batch_size"]
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(training["epochs"]):
        start = time.time()
        model.train()
        losses = []
        for b in make_loader(ds["train"], bs, True, seed + epoch, drop_last=True):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(*_inputs(b, tp_col)), b["fut"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val = float(np.mean([F.mse_loss(model(*_inputs(b, tp_col)), b["fut"].to(DEVICE)).item()
                                 for b in make_loader(ds["val"], bs, False, 0)]))
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_loss=val,
                            seconds=round(time.time() - start, 1)))
        print(f"epoch {epoch + 1:3d}  train {np.mean(losses):.5f}  val {val:.5f}", flush=True)
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= training["patience"]:
                break
    model.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    preds = {split: predict_vanilla(model, ds[split], tp_col, bs) for split in ("val", "test")}
    return preds, history
