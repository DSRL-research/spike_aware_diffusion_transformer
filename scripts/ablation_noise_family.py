import json
from pathlib import Path

from configs import heat
from configs.base import NoiseConfig
from src.datasets import SeriesData
from src.eval.metrics import summarize
from src.eval.significance import paired_tests, window_errors
from src.integration import s1_shift
from src.models.spike_probs import SpikeProbTable
from src.pipeline import (RESULTS, load_predictions, probabilities_path, run_forecaster, save_predictions,
                          score_split, write_json)

SEEDS = (42, 43, 44)
FAMILIES = ("gaussian", "student_t")
MODELS = ("diffusion", "s1", "s2", "s3")
LABELS = {"johnson_su": "Johnson-SU", "gaussian": "Gaussian", "student_t": "Student-t"}


def model_dir(fam: str, model: str, seed: int) -> Path:
    if fam == "johnson_su":
        return RESULTS / "heat" / model / f"seed_{seed}"
    return RESULTS / "heat" / f"noise_{fam}" / model / f"seed_{seed}"


def noise_config(fam: str) -> NoiseConfig:
    params = json.loads((RESULTS / "heat" / "noise_families.json").read_text())[fam]
    return NoiseConfig(family=fam, student_df=params.get("student_df", 5), base=dict(params["base"]),
                       clamp_a=tuple(params["clamp_a"]), clamp_b=tuple(params["clamp_b"]),
                       clamp_loc=tuple(params["clamp_loc"]), clamp_scale=tuple(params["clamp_scale"]))


def train() -> None:
    kappa, beta = heat.S1
    table = SpikeProbTable(probabilities_path("heat"), SeriesData(heat.DATA)).probs
    for fam in FAMILIES:
        for seed in SEEDS:
            for model in ("diffusion", "s2", "s3"):
                out = model_dir(fam, model, seed)
                if not (out / "predictions_test.npz").exists():
                    run_forecaster(heat.forecaster(model, noise_config(fam)), seed, out, probabilities_path("heat"),
                                   eq32=heat.EQ32.get(model, 0.0))
            for split in ("val", "test"):
                pred = load_predictions(model_dir(fam, "diffusion", seed) / f"predictions_{split}.npz")
                pred["pred_norm"] = pred["pred_norm"] + s1_shift(table[pred["window"]], kappa, beta)
                save_predictions(model_dir(fam, "s1", seed) / f"predictions_{split}.npz", pred)


def main():
    train()
    unit, scale = heat.UNIT
    preds = {(fam, m): [load_predictions(model_dir(fam, m, s) / "predictions_test.npz") for s in SEEDS]
             for fam in ("johnson_su",) + FAMILIES for m in MODELS}
    errors, hist_end, has_spike = {}, None, None
    for key, ps in preds.items():
        errors[key], hist_end, has_spike = window_errors(ps)
    pairs = [((fam, m), ("johnson_su", m)) for fam in FAMILIES for m in MODELS]
    tests = paired_tests(errors, pairs, hist_end, has_spike, scale)
    lines = [f"# Heat demand: noise families, test, seeds {', '.join(map(str, SEEDS))} ({unit}); r = rank-biserial "
             "correlation against Johnson-SU, * Holm p < 0.05 (all windows), † Holm p < 0.05 (non-overlapping "
             "windows)", "",
             "| model | noise | MAE | RMSE | R² | Spike-MAE | Spike-RMSE |", "|---|---|---|---|---|---|---|"]
    out = {}
    for m in MODELS:
        for fam in ("johnson_su",) + FAMILIES:
            per_seed = [{k: v * (1 if k in ("r2", "smape") else scale) for k, v in score_split(heat, p).items()}
                        for p in preds[(fam, m)]]
            s = summarize(per_seed)
            out[f"{m}|{fam}"] = s
            cells = []
            for k, metric in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", None), ("spike_mae", "Spike-MAE"),
                              ("spike_rmse", "Spike-RMSE")):
                cell = f"{s[k]['mean']:.{4 if k == 'r2' else 1}f}±{s[k]['std']:.{4 if k == 'r2' else 1}f}"
                if fam != "johnson_su" and metric:
                    ta = tests[("all", (fam, m), ("johnson_su", m), metric)]
                    tn = tests[("non-overlapping", (fam, m), ("johnson_su", m), metric)]
                    cell += f" r = {ta['r']:+.2f}" + ("*" if ta["p_holm"] < 0.05 else "")
                    cell += "†" if tn["p_holm"] < 0.05 else ""
                cells.append(cell)
            lines.append(f"| {m} | {LABELS[fam]} | " + " | ".join(cells) + " |")
    write_json(RESULTS / "heat" / "noise_family.json",
               dict(metrics=out, tests={"|".join(map(str, k)): v for k, v in tests.items()}))
    (RESULTS / "heat" / "noise_family.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
