import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configs.base import ROOT

PTS_DIR = ROOT / "third_party" / "pytorch-ts"
GLUONTS_DIR = ROOT / "third_party" / "gluonts_v0.9.0" / "src"
KNOWN_LAGS = ("tgt_level__lag_24", "tgt_level__lag_168", "tgt_level__same_hour_mean_7d")
LAGS_SEQ = [1, 24, 168]
NETWORK = dict(num_layers=2, num_cells=40, cell_type="GRU", dropout_rate=0.1, conditioning_length=100, diff_steps=100,
               loss_type="l2", beta_end=0.1, beta_schedule="linear", residual_layers=8, residual_channels=8,
               dilation_cycle_length=2, scaling=True)
TRAINER = dict(epochs=100, batch_size=64, num_batches_per_epoch=100, learning_rate=1e-3, weight_decay=1e-6,
               maximum_learning_rate=1e-2, clip_gradient=None)
PATIENCE = 10
NUM_SAMPLES = 100


def _package(name: str, path: Path = None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, mod)
    return sys.modules[name]


def timegrad_network():
    if "pts.model.time_grad.time_grad_network" in sys.modules:
        return sys.modules["pts.model.time_grad.time_grad_network"]
    _package("gluonts", GLUONTS_DIR / "gluonts")
    _package("gluonts.core")
    component = _package("gluonts.core.component")
    component.validated = lambda *args, **kwargs: (lambda init: init)
    _package("gluonts.torch", GLUONTS_DIR / "gluonts" / "torch")
    _package("gluonts.torch.modules", GLUONTS_DIR / "gluonts" / "torch" / "modules")
    _package("pts", PTS_DIR / "pts")
    model = _package("pts.model", PTS_DIR / "pts" / "model")
    modules = _package("pts.modules", PTS_DIR / "pts" / "modules")
    _package("pts.model.time_grad", PTS_DIR / "pts" / "model" / "time_grad")
    model.weighted_average = importlib.import_module("pts.model.utils").weighted_average
    for sub in ("gaussian_diffusion", "scaler", "distribution_output"):
        mod = importlib.import_module(f"pts.modules.{sub}")
        for name in ("GaussianDiffusion", "MeanScaler", "NOPScaler", "DiffusionOutput"):
            if hasattr(mod, name):
                setattr(modules, name, getattr(mod, name))
    return importlib.import_module("pts.model.time_grad.time_grad_network")


class TimeGradWindows(Dataset):
    def __init__(self, base, training: bool):
        self.base, self.training = base, training
        s, cfg = base.series, base.cfg
        names = s.hist_feature_names
        known = set(s.cal_cols) | set(cfg.future_known) | set(KNOWN_LAGS)
        self.time_features = [n for n in names if n in known]
        self.target_dims = ["target"] + [n for n in names if n not in known]
        self.context_length = cfg.history
        self.history_length = self.context_length + max(LAGS_SEQ)
        self._time_idx = [names.index(n) for n in self.time_features]
        self._dim_idx = [names.index(n) for n in self.target_dims[1:]]

    @property
    def input_size(self) -> int:
        return len(self.target_dims) * (len(LAGS_SEQ) + 1) + len(self.time_features)

    def __len__(self):
        return len(self.base)

    def features(self, rows: np.ndarray, mean: float, std: float, values: np.ndarray) -> np.ndarray:
        s = self.base.series
        parts = [s.cal[rows]]
        if s.obs_cols:
            parts.append(s.obs[rows])
        parts.append(((s.level[rows] - mean) / std).astype(np.float32))
        parts.append((s.scale[rows] / std).astype(np.float32))
        if s.cfg.absolute_level:
            mu, sd = s.level_stats
            parts.append(((values.astype(np.float64) - mu) / sd).astype(np.float32)[:, None])
        return np.concatenate(parts, axis=1)

    def __getitem__(self, i):
        it = self.base[i]
        s, cfg = self.base.series, self.base.cfg
        he = int(s.windows["hist_end"].iloc[int(it["window"])])
        mean, std = float(it["mean"]), float(it["std"])
        P, F = self.history_length, cfg.horizon
        past = np.arange(he - P, he)
        pad = past < 0
        past = np.maximum(past, 0)
        values = s.history_values[past]
        feat = self.features(past, mean, std, values)
        target = np.concatenate([((values - mean) / std)[:, None], feat[:, self._dim_idx]], axis=1)
        target[pad] = 0.0
        time_feat = feat[:, self._time_idx]
        time_feat[pad] = 0.0
        fut = np.arange(he, he + F)
        ffeat = self.features(fut, mean, std, s.target[fut])
        D = len(self.target_dims)
        item = {
            "target_dimension_indicator": torch.arange(D),
            "past_time_feat": torch.from_numpy(time_feat.astype(np.float32)),
            "past_target_cdf": torch.from_numpy(target.astype(np.float32)),
            "past_observed_values": torch.ones(P, D),
            "past_is_pad": torch.from_numpy(pad.astype(np.float32)),
            "future_time_feat": torch.from_numpy(ffeat[:, self._time_idx].astype(np.float32)),
        }
        if self.training:
            item["future_target_cdf"] = torch.from_numpy(
                np.concatenate([it["fut"].numpy(), ffeat[:, self._dim_idx]], axis=1).astype(np.float32))
            item["future_observed_values"] = torch.ones(F, D)
        else:
            for k in ("window", "mean", "std", "fut_raw", "label"):
                item[k] = it[k]
        return item


def network_kwargs(dataset: TimeGradWindows) -> dict:
    return dict(NETWORK, input_size=dataset.input_size, target_dim=len(dataset.target_dims),
                history_length=dataset.history_length, context_length=dataset.context_length,
                prediction_length=dataset.base.cfg.horizon, lags_seq=list(LAGS_SEQ))


@torch.no_grad()
def predict(net, dataset: TimeGradWindows, windows_per_pass: int, device) -> dict:
    net.eval()
    names = ("target_dimension_indicator", "past_time_feat", "past_target_cdf", "past_observed_values", "past_is_pad",
             "future_time_feat")
    out = {k: [] for k in ("samples", "y_true", "spike", "mean", "std", "window")}
    for batch in DataLoader(dataset, batch_size=windows_per_pass, shuffle=False):
        samples = net(*[batch[k].to(device) for k in names])
        out["samples"].append(samples[..., 0].cpu().numpy())
        out["y_true"].append(batch["fut_raw"].numpy())
        out["spike"].append(batch["label"][:, :, 0].bool().numpy())
        for k in ("mean", "std", "window"):
            out[k].append(batch[k].numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["pred_norm"] = np.median(res.pop("samples"), axis=1)
    res["hist_end"] = dataset.base.series.windows["hist_end"].to_numpy()[res["window"]]
    return res
