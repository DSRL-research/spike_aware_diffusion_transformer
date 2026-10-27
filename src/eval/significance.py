import numpy as np
from scipy.stats import wilcoxon

METRICS = ("MAE", "RMSE", "Spike-MAE", "Spike-RMSE")


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = running
    return out


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def window_errors(preds: list, clip: bool = False) -> tuple:
    per, hist_end, spike = [], None, None
    for pred in preds:
        pn = pred["pred_norm"]
        if clip:
            pn = np.maximum(pn, (-pred["mean"] / pred["std"])[:, None])
        f = pn * pred["std"][:, None] + pred["mean"][:, None]
        o = np.argsort(pred["window"])
        e, sp = (f - pred["y_true"])[o], pred["spike"][o].astype(bool)
        if hist_end is None:
            hist_end, spike = pred["hist_end"][o], sp
        elif not (np.array_equal(hist_end, pred["hist_end"][o]) and np.array_equal(spike, sp)):
            raise ValueError("predictions with other test windows or labels")
        n = np.maximum(sp.sum(axis=1), 1)
        per.append(np.stack([np.abs(e).mean(axis=1), np.sqrt((e ** 2).mean(axis=1)),
                             (np.abs(e) * sp).sum(axis=1) / n, np.sqrt((e ** 2 * sp).sum(axis=1) / n)]))
    return np.mean(per, axis=0), hist_end, spike.any(axis=1)


def paired_tests(errors: dict, pairs: list, hist_end: np.ndarray, has_spike: np.ndarray, scale: float = 1.0) -> dict:
    nonoverlapping = (hist_end - hist_end[0]) % 24 == 0
    tests = {}
    for subset, base_mask in (("all", np.ones(len(hist_end), bool)), ("non-overlapping", nonoverlapping)):
        keys = []
        for a_name, b_name in pairs:
            for mi, metric in enumerate(METRICS):
                mask = base_mask & has_spike if metric.startswith("Spike") else base_mask
                a, b = errors[a_name][mi][mask] * scale, errors[b_name][mi][mask] * scale
                d = a - b
                key = (subset, a_name, b_name, metric)
                tests[key] = dict(p=1.0 if not np.any(d) else float(wilcoxon(a, b).pvalue), delta=float(d.mean()),
                                  r=rank_biserial(a, b), n=int(mask.sum()))
                keys.append(key)
        for k, ph in holm({k: tests[k]["p"] for k in keys}).items():
            tests[k]["p_holm"] = ph
    return tests
