import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from configs.base import ROOT, DataConfig
from src.features import feature_columns
from src.seed import loader_generator, worker_init_fn
from src.splits import split_indices

STD_EPS = 1e-6
SPLITS = ("train", "val", "test")


def window_stats(hist: np.ndarray) -> tuple:
    return float(hist.mean()), float(hist.std() + STD_EPS)


def make_windows(index: pd.DatetimeIndex, history: int, horizon: int) -> pd.DataFrame:
    ends = np.arange(history, len(index) - horizon)
    ts = index[ends]
    return pd.DataFrame({"timestamp": ts, "hist_end": ends, "forecast_hour": ts.hour, "forecast_weekday": ts.weekday,
                         "forecast_month": ts.month})


class SeriesData:
    def __init__(self, cfg: DataConfig):
        path = ROOT / cfg.path
        names = pq.read_schema(path).names
        feat_cols = feature_columns(names, cfg.feature_groups)
        df = pd.read_parquet(path, columns=["target", "target_valid", "target_filled"] + feat_cols)
        if cfg.normalization not in ("window", "global"):
            raise ValueError(f"Unknown normalization '{cfg.normalization}'")
        self.cfg = cfg
        self.index = df.index
        self.target = df["target"].to_numpy(dtype=np.float32)
        self.history_values = df["target_filled"].to_numpy(dtype=np.float32)
        self.valid = df["target_valid"].to_numpy(dtype=bool)
        self.cal_cols = [c for c in feat_cols if c.startswith("cal__")]
        self.obs_cols = [c for c in feat_cols if c.startswith("obs__")]
        self.level_cols = [c for c in feat_cols if c.startswith("tgt_level__")]
        self.scale_cols = [c for c in feat_cols if c.startswith("tgt_scale__")]
        self.cal = df[self.cal_cols].to_numpy(dtype=np.float32)
        missing = [c for c in cfg.future_known if c not in self.obs_cols]
        if missing:
            raise ValueError(f"future_known {missing} are not obs__ columns of {cfg.path}")
        self.future_idx = [self.obs_cols.index(c) for c in cfg.future_known]
        self.obs_raw = df[self.obs_cols].to_numpy(dtype=np.float64)
        self.obs = None
        self.level_stats = None
        self.scaler_key = None
        self.level = df[self.level_cols].to_numpy(dtype=np.float64)
        self.scale = df[self.scale_cols].to_numpy(dtype=np.float64)
        self.windows = make_windows(df.index, cfg.history, cfg.horizon)
        ends = self.windows["hist_end"].to_numpy()
        self.window_valid = np.array([self.valid[e:e + cfg.horizon].all() for e in ends])

    @property
    def hist_feature_names(self) -> list:
        absolute = ["abs__level"] if self.cfg.absolute_level else []
        return self.cal_cols + self.obs_cols + self.level_cols + self.scale_cols + absolute

    @property
    def fut_feature_names(self) -> list:
        return list(self.cal_cols) + [f"future__{c}" for c in self.cfg.future_known]

    @property
    def context_dim(self) -> int:
        return len(self.hist_feature_names) + len(self.cal_cols) + 5

    def fit_scalers(self, rows: np.ndarray) -> None:
        key = np.flatnonzero(rows).tobytes()
        if self.scaler_key == key:
            return
        if self.obs_cols:
            train = self.obs_raw[rows]
            mean, std = train.mean(axis=0), train.std(axis=0)
            self.obs = ((self.obs_raw - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
        values = self.history_values[rows].astype(np.float64)
        self.level_stats = (values.mean(), values.std())
        self.scaler_key = key

    def training_rows(self, idx: dict) -> np.ndarray:
        cfg = self.cfg
        ends = self.windows["hist_end"].to_numpy()
        diff = np.zeros(len(self.index) + 1, dtype=np.int64)
        np.add.at(diff, ends[idx["train"]] - cfg.history, 1)
        np.add.at(diff, ends[idx["train"]] + cfg.horizon, -1)
        rows = np.cumsum(diff)[:-1] > 0
        for split, w in idx.items():
            if split != "train" and len(w):
                diff = np.zeros(len(self.index) + 1, dtype=np.int64)
                np.add.at(diff, ends[w], 1)
                np.add.at(diff, ends[w] + cfg.horizon, -1)
                rows &= ~(np.cumsum(diff)[:-1] > 0)
        return rows


class WindowDataset(Dataset):
    def __init__(self, series: SeriesData, indices: np.ndarray, cfg: DataConfig = None):
        self.series = series
        self.cfg = cfg or series.cfg
        self.indices = np.asarray(indices)
        w = series.windows
        self._hist_end = w["hist_end"].to_numpy()
        self._hour = w["forecast_hour"].to_numpy()
        self._weekday = w["forecast_weekday"].to_numpy()
        self._month = w["forecast_month"].to_numpy()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s, cfg = self.series, self.cfg
        w = int(self.indices[i])
        he = int(self._hist_end[w])
        hs, fe = he - cfg.history, he + cfg.horizon
        hist = s.history_values[hs:he]
        fut = s.target[he:fe]
        w_mean, w_std = window_stats(hist)
        if s.cfg.normalization == "global":
            mean, std = float(np.float32(s.level_stats[0])), float(np.float32(s.level_stats[1]))
        else:
            mean, std = w_mean, w_std
        hist_n = (hist - mean) / std
        fut_n = (fut - mean) / std
        label = (fut - w_mean) / w_std > cfg.spike_k
        hour = int(self._hour[w])
        parts = [s.cal[hs:he]]
        if s.obs_cols:
            parts.append(s.obs[hs:he])
        parts.append(((s.level[hs:he] - mean) / std).astype(np.float32))
        parts.append((s.scale[hs:he] / std).astype(np.float32))
        if s.cfg.absolute_level:
            mu, sd = s.level_stats
            parts.append(((hist.astype(np.float64) - mu) / sd).astype(np.float32)[:, None])
        hist_feat = np.concatenate(parts, axis=1)
        fut_feat = s.cal[he:fe]
        if s.future_idx:
            fut_feat = np.concatenate([fut_feat, s.obs[he:fe][:, s.future_idx]], axis=1)
        mu, sd = s.level_stats
        extra = [hour / 24.0, self._weekday[w] / 7.0, self._month[w] / 12.0, (mean - mu) / sd, std / sd]
        context = np.concatenate([hist_feat.mean(axis=0, dtype=np.float64), s.cal[he], extra])
        return {
            "hist": torch.from_numpy(hist_n[:, None]),
            "fut": torch.from_numpy(fut_n[:, None]),
            "hour": torch.tensor(hour, dtype=torch.long),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "label": torch.from_numpy(label[:, None].astype(np.float32)),
            "fut_raw": torch.from_numpy(fut.copy()),
            "window": torch.tensor(w, dtype=torch.long),
            "hist_feat": torch.from_numpy(hist_feat),
            "fut_feat": torch.from_numpy(fut_feat.copy()),
            "context": torch.from_numpy(context.astype(np.float32)),
        }


def make_datasets(cfg: DataConfig, series: SeriesData = None) -> dict:
    series = series or SeriesData(cfg)
    for name in ("path", "history", "horizon", "feature_groups", "absolute_level", "normalization", "future_known"):
        if getattr(series.cfg, name) != getattr(cfg, name):
            raise ValueError(f"series was built with a different data config ({name})")
    idx = split_indices(len(series.windows), cfg.split, cfg.split_gap)
    series.fit_scalers(series.training_rows(idx))
    idx = {k: v[series.window_valid[v]] for k, v in idx.items()}
    return {s: WindowDataset(series, idx[s], cfg) for s in SPLITS}


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, drop_last: bool = False,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
                      generator=loader_generator(seed), worker_init_fn=worker_init_fn)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}
