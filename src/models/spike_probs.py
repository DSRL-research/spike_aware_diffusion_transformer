from pathlib import Path

import numpy as np
import torch


class SpikeProbTable:
    def __init__(self, path: Path, series=None):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.probs = 1.0 / (1.0 + np.exp(-z["logits"].astype(np.float64)))
        if series is not None:
            times = series.index[series.windows["hist_end"].to_numpy()].asi8
            if len(times) != len(z["hist_end_time"]) or not np.array_equal(times, z["hist_end_time"]):
                raise ValueError(f"{path}: windows do not match the forecaster's series")
        self._cache = {}

    def table(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._cache:
            self._cache[key] = torch.tensor(np.nan_to_num(self.probs, nan=-1.0), dtype=torch.float32, device=device)
        return self._cache[key]

    def __call__(self, batch: dict) -> torch.Tensor:
        p = self.table(batch["window"].device)[batch["window"]].unsqueeze(-1)
        if (p < 0).any():
            raise RuntimeError(f"{self.path}: no probabilities for some windows of this batch")
        return p
