import numpy as np
import torch

from configs.base import ExperimentConfig
from src.datasets import WindowDataset, make_loader, to_device
from src.eval.sampler import ddim_sample
from src.seed import window_generators

BATCH_SIZE = 32


@torch.no_grad()
def predict(cfg: ExperimentConfig, dataset: WindowDataset, model, noise=None, probs=None, seed: int = 0) -> dict:
    device = next(model.parameters()).device
    model.eval()
    if noise is not None:
        noise.eval()
    out = {k: [] for k in ("pred_norm", "y_true", "spike", "mean", "std", "window")}
    for batch in make_loader(dataset, BATCH_SIZE, shuffle=False, seed=seed):
        batch = to_device(batch, device)
        if cfg.model_type == "diffusion":
            p_t = probs(batch) if probs is not None else None
            gens = window_generators(seed, batch["window"].tolist(), device)
            pred = ddim_sample(model, noise, batch, cfg.diffusion, cfg.data.horizon, p_t=p_t, generator=gens)
        else:
            pred = model(batch["hist"], batch["hist_feat"], batch["fut_feat"])
        out["pred_norm"].append(pred.squeeze(-1).cpu().numpy())
        out["y_true"].append(batch["fut_raw"].cpu().numpy())
        out["spike"].append(batch["label"].squeeze(-1).bool().cpu().numpy())
        out["mean"].append(batch["mean"].cpu().numpy())
        out["std"].append(batch["std"].cpu().numpy())
        out["window"].append(batch["window"].cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["hist_end"] = dataset.series.windows["hist_end"].to_numpy()[res["window"]]
    return res
