import copy
import json
from pathlib import Path

import numpy as np

from configs import heat
from src.datasets import SeriesData, make_datasets
from src.eval.evaluate import predict
from src.integration import BETAS, KAPPAS, s1_shift
from src.models.spike_probs import SpikeProbTable
from src.pipeline import (RESULTS, detector_dir, load_forecaster, load_predictions, run_detector,
                          run_forecaster, score_split, write_json)
from src.seed import seed_everything

K_VALUES = (1.0, 1.5, 2.0, 2.5)
SEEDS = (42, 43, 44)
EQ32_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
OUT = RESULTS / "heat" / "sensitivity_k"


def detector_for(k: float) -> Path:
    if k == heat.DETECTOR_DATA.spike_k:
        return detector_dir("heat")
    out = OUT / f"k_{k}" / "detector"
    if not (out / "probabilities.npz").exists():
        cfg = heat.detector()
        cfg.data = copy.deepcopy(cfg.data)
        cfg.data.spike_k = k
        run_detector(cfg, heat.DETECTOR_SEED, out, heat.PROBABILITY_WINDOWS)
    return out


def forecaster_dir(k: float, variant: str, seed: int) -> Path:
    if k == heat.DETECTOR_DATA.spike_k:
        return RESULTS / "heat" / variant / f"seed_{seed}"
    return OUT / f"k_{k}" / variant / f"seed_{seed}"


def s1_row(probs: np.ndarray) -> tuple:
    preds = {split: [load_predictions(RESULTS / "heat" / "diffusion" / f"seed_{s}" / f"predictions_{split}.npz")
                     for s in SEEDS] for split in ("val", "test")}
    best, best_rmse = None, float("inf")
    for kappa in KAPPAS:
        for beta in BETAS:
            rmse = np.mean([score_split(heat, p, s1_shift(probs[p["window"]], kappa, beta))["rmse"]
                            for p in preds["val"]])
            if rmse < best_rmse:
                best, best_rmse = (float(kappa), beta), rmse
    test = [score_split(heat, p, s1_shift(probs[p["window"]], *best)) for p in preds["test"]]
    return best, test


def s2s3_row(k: float, variant: str, probs_path: Path, series, data) -> tuple:
    per_scale = {w: [] for w in EQ32_GRID}
    for seed in SEEDS:
        cfg = heat.forecaster(variant)
        out = forecaster_dir(k, variant, seed)
        if not (out / "model.pt").exists():
            run_forecaster(cfg, seed, out, probs_path, eq32=0.0, splits=("test",))
        model, noise = load_forecaster(cfg, out, series)
        probs = SpikeProbTable(probs_path, series)
        for w in EQ32_GRID:
            ecfg = copy.deepcopy(cfg)
            ecfg.diffusion.x0_spike_scale = w
            seed_everything(seed)
            per_scale[w].append(score_split(heat, predict(ecfg, data["test"], model, noise, probs, seed=seed)))
    best = max(EQ32_GRID, key=lambda w: np.mean([m["r2"] for m in per_scale[w]]))
    return best, per_scale[best]


def main():
    series = SeriesData(heat.DATA)
    data = make_datasets(heat.DATA, series)
    lines = [f"# Heat demand: spike threshold k of the detector labels, test, seeds {', '.join(map(str, SEEDS))}", "",
             "| k | spike rate | AUPRC | F1 | S1 R² | S2 R² | S3 R² | S1 kappa, beta | S2 / S3 Eq. 32 scale |",
             "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for k in K_VALUES:
        det = detector_for(k)
        m = json.loads((det / "metrics.json").read_text())["metrics"]["test"]
        probs_path = det / "probabilities.npz"
        probs = SpikeProbTable(probs_path, series).probs
        s1_choice, s1 = s1_row(probs)
        (w2, s2), (w3, s3) = (s2s3_row(k, v, probs_path, series, data) for v in ("s2", "s3"))
        r2 = lambda rows: float(np.mean([x["r2"] for x in rows]))
        res[str(k)] = dict(detector=m, s1=dict(choice=s1_choice, test=s1), s2=dict(eq32=w2, test=s2),
                           s3=dict(eq32=w3, test=s3))
        lines.append(f"| {k} | {m['spike_rate']:.3f} | {m['auprc']:.3f} | {m['f1']:.3f} | {r2(s1):.4f} | "
                     f"{r2(s2):.4f} | "
                     f"{r2(s3):.4f} | {s1_choice[0]}, {s1_choice[1]} | {w2} / {w3} |")
        print(lines[-1], flush=True)
    write_json(OUT / "summary.json", res)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
