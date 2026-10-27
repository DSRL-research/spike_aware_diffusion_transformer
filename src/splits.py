import numpy as np


def split_indices(n_windows: int, fractions: tuple, gap: int) -> dict:
    train_f, val_f, test_f = fractions
    if abs(train_f + val_f + test_f - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    b1 = int(round(train_f * n_windows))
    b2 = int(round(round(train_f + val_f, 10) * n_windows))
    return {
        "train": np.arange(0, b1),
        "val": np.arange(min(b1 + gap, b2), b2),
        "test": np.arange(min(b2 + gap, n_windows), n_windows),
    }
