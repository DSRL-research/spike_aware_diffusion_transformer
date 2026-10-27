import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from configs.base import ROOT

CSDI_DIR = ROOT / "third_party" / "CSDI"
KNOWN_LAGS = ("tgt_level__lag_24", "tgt_level__lag_168", "tgt_level__same_hour_mean_7d")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def csdi_modules():
    if "linear_attention_transformer" not in sys.modules:
        stub = types.ModuleType("linear_attention_transformer")

        def missing(*args, **kwargs):
            raise ImportError("linear_attention_transformer is not installed: use diffusion.is_linear: False")
        stub.LinearAttentionTransformer = missing
        sys.modules["linear_attention_transformer"] = stub
    _load("diff_models", CSDI_DIR / "diff_models.py")
    return _load("csdi_main_model", CSDI_DIR / "main_model.py"), _load("csdi_utils", CSDI_DIR / "utils.py")


def csdi_config(overrides: dict = None) -> dict:
    cfg = yaml.safe_load((CSDI_DIR / "config" / "base_forecasting.yaml").read_text())
    cfg["model"]["is_unconditional"] = 0
    cfg["diffusion"]["is_linear"] = False
    for key, value in (overrides or {}).items():
        section, name = key.split(".")
        cfg[section][name] = value
    return cfg


class CSDIWindows(Dataset):
    def __init__(self, base, known_lags: bool = False):
        self.base = base
        s = base.series
        hist_names, fut_names = s.hist_feature_names, s.fut_feature_names
        self.lag_map = ([(hist_names.index(n), s.level_cols.index(n)) for n in KNOWN_LAGS if n in hist_names]
                        if known_lags else [])
        self.future_map = [(hist_names.index(n[len("future__"):] if n.startswith("future__") else n), j)
                           for j, n in enumerate(fut_names)]
        self.channels = ["target"] + list(hist_names)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base[i]
        hist, fut = it["hist"][:, 0], it["fut"][:, 0]
        hf, ff = it["hist_feat"], it["fut_feat"]
        H, F, C = len(hist), len(fut), hf.shape[1]
        data = torch.zeros(H + F, 1 + C)
        observed = torch.zeros_like(data)
        given = torch.zeros_like(data)
        data[:H, 0], data[H:, 0] = hist, fut
        observed[:, 0] = 1.0
        given[:H, 0] = 1.0
        data[:H, 1:] = hf
        observed[:H, 1:] = 1.0
        given[:H, 1:] = 1.0
        for c, j in self.future_map:
            data[H:, 1 + c] = ff[:, j]
            observed[H:, 1 + c] = 1.0
            given[H:, 1 + c] = 1.0
        if self.lag_map:
            s = self.base.series
            he = int(s.windows["hist_end"].iloc[int(it["window"])])
            mean, std = float(it["mean"]), float(it["std"])
            for c, j in self.lag_map:
                data[H:, 1 + c] = torch.from_numpy(((s.level[he:he + F, j] - mean) / std).astype(np.float32))
                observed[H:, 1 + c] = 1.0
                given[H:, 1 + c] = 1.0
        return {
            "observed_data": data, "observed_mask": observed, "gt_mask": given,
            "timepoints": torch.arange(H + F, dtype=torch.float32),
            "feature_id": torch.arange(1 + C, dtype=torch.float32),
            "window": it["window"], "mean": it["mean"], "std": it["std"], "fut_raw": it["fut_raw"],
            "label": it["label"][:, 0],
        }


@torch.no_grad()
def predict(model, dataset: CSDIWindows, n_samples: int, rows_per_pass: int) -> dict:
    model.eval()
    model.target_dim = model.target_dim_base
    H = dataset.base.cfg.history
    per_pass = max(1, rows_per_pass // n_samples)
    out = {k: [] for k in ("samples", "y_true", "spike", "mean", "std", "window")}
    for batch in DataLoader(dataset, batch_size=per_pass, shuffle=False):
        observed_data, observed_mask, observed_tp, gt_mask, _, _, _ = model.process_data(batch)
        B = observed_data.shape[0]
        rep = lambda x: x.repeat_interleave(n_samples, dim=0)
        cond_mask = rep(gt_mask)
        side_info = model.get_side_info(rep(observed_tp), cond_mask)
        draws = model.impute(rep(observed_data), cond_mask, side_info, 1)
        out["samples"].append(draws[:, 0, 0, H:].reshape(B, n_samples, -1).cpu().numpy())
        out["y_true"].append(batch["fut_raw"].numpy())
        out["spike"].append(batch["label"].bool().numpy())
        for k in ("mean", "std", "window"):
            out[k].append(batch[k].numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["pred_norm"] = np.median(res.pop("samples"), axis=1)
    res["hist_end"] = dataset.base.series.windows["hist_end"].to_numpy()[res["window"]]
    return res
