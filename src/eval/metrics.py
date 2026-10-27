import numpy as np

SMAPE_EPS = 1e-6
KEYS = ("mae", "rmse", "r2", "smape", "spike_mae", "spike_rmse")


def point_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return dict(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        smape=float(100 * np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + SMAPE_EPS))),
    )


def spike_metrics(y_true, y_pred, mask) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return dict(spike_mae=float("nan"), spike_rmse=float("nan"))
    err = np.asarray(y_pred, dtype=np.float64)[mask] - np.asarray(y_true, dtype=np.float64)[mask]
    return dict(spike_mae=float(np.mean(np.abs(err))), spike_rmse=float(np.sqrt(np.mean(err ** 2))))


def overlap_average(hist_end, values, n_total: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = np.zeros(n_total)
    count = np.zeros(n_total)
    F = values.shape[1]
    for he, row in zip(np.asarray(hist_end), values):
        total[he:he + F] += row
        count[he:he + F] += 1
    covered = np.where(count > 0)[0]
    return total[covered] / count[covered]


def overlap_metrics(hist_end, y_true, y_pred, spike) -> dict:
    n_total = int(np.asarray(hist_end).max()) + np.asarray(y_pred).shape[1]
    p = overlap_average(hist_end, y_pred, n_total)
    y = overlap_average(hist_end, y_true, n_total)
    share = overlap_average(hist_end, np.asarray(spike, dtype=np.float64), n_total)
    return {**point_metrics(y, p), **spike_metrics(y, p, share >= 0.5)}


def forecast(pred: dict, shift=None, clip: bool = False) -> np.ndarray:
    pn = pred["pred_norm"] if shift is None else pred["pred_norm"] + shift
    if clip:
        pn = np.maximum(pn, (-pred["mean"] / pred["std"])[:, None])
    return pn * pred["std"][:, None] + pred["mean"][:, None]


def scores(pred: dict, shift=None, clip: bool = False) -> dict:
    return overlap_metrics(pred["hist_end"], pred["y_true"], forecast(pred, shift, clip), pred["spike"])


def summarize(per_seed: list) -> dict:
    return {k: dict(mean=float(np.mean([m[k] for m in per_seed])), std=float(np.std([m[k] for m in per_seed], ddof=1))
                    if len(per_seed) > 1 else float("nan")) for k in KEYS}


def format_row(name: str, per_seed: list) -> str:
    s = summarize(per_seed)
    cells = [f"{s[k]['mean']:.{4 if k == 'r2' else 2}f} ± {s[k]['std']:.{4 if k == 'r2' else 2}f}" for k in KEYS]
    return f"| {name} | " + " | ".join(cells) + " |"


def table_header() -> list:
    return ["| model | MAE | RMSE | R² | sMAPE % | Spike-MAE | Spike-RMSE |", "|---|---|---|---|---|---|---|"]
