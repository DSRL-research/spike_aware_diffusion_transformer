import argparse
from pathlib import Path

from configs import DATASETS, dataset_module
from src.datasets import SeriesData
from src.eval.metrics import KEYS, format_row, table_header
from src.integration import s1_shift
from src.models.spike_probs import SpikeProbTable
from src.pipeline import (RESULTS, load_predictions, probabilities_path, run_forecaster, save_predictions,
                          score_split, write_json)
from src.train.train_vanilla import run_vanilla

MODELS = ("transformer", "diffusion", "s2", "s3")
NAMES = {"transformer": "Transformer", "diffusion": "Diffusion only", "s1": "S1", "s2": "S2", "s3": "S3"}


def model_dir(dataset: str, model: str, seed: int) -> Path:
    return RESULTS / dataset / model / f"seed_{seed}"


def train(dataset: str, model: str, seed: int) -> None:
    ds = dataset_module(dataset)
    out = model_dir(dataset, model, seed)
    if (out / "predictions_test.npz").exists():
        return
    print(f"{dataset} {model} seed {seed}", flush=True)
    if model == "transformer" and dataset == "wind":
        preds, history = run_vanilla(ds.VANILLA_DATA, ds.VANILLA_TRAINING, seed, out)
        write_json(out / "history.json", history)
        for split, pred in preds.items():
            save_predictions(out / f"predictions_{split}.npz", pred)
    elif model == "transformer":
        run_forecaster(ds.transformer(), seed, out)
    else:
        run_forecaster(ds.forecaster(model), seed, out, probabilities_path(dataset), eq32=ds.EQ32.get(model, 0.0))


def apply_s1(dataset: str, seed: int) -> None:
    ds = dataset_module(dataset)
    kappa, beta = ds.S1
    table = SpikeProbTable(probabilities_path(dataset), SeriesData(ds.DATA)).probs
    for split in ("val", "test"):
        pred = load_predictions(model_dir(dataset, "diffusion", seed) / f"predictions_{split}.npz")
        pred["pred_norm"] = pred["pred_norm"] + s1_shift(table[pred["window"]], kappa, beta)
        save_predictions(model_dir(dataset, "s1", seed) / f"predictions_{split}.npz", pred)


def summary(dataset: str, seeds) -> None:
    ds = dataset_module(dataset)
    unit, scale = ds.UNIT
    rows, lines = {}, [f"# {dataset}: test split, overlap-averaged, mean ± std over seeds {', '.join(map(str, seeds))} "
                       f"({unit})", ""] + table_header()
    for model in ("transformer", "diffusion", "s1", "s2", "s3"):
        per_seed = []
        for seed in seeds:
            m = score_split(ds, load_predictions(model_dir(dataset, model, seed) / "predictions_test.npz"))
            per_seed.append({k: m[k] * (1 if k in ("r2", "smape") else scale) for k in KEYS})
        rows[model] = per_seed
        lines.append(format_row(NAMES[model], per_seed))
    write_json(RESULTS / dataset / "summary.json", rows)
    (RESULTS / dataset / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    seeds = args.seeds or dataset_module(args.dataset).SEEDS
    for seed in seeds:
        for model in MODELS:
            train(args.dataset, model, seed)
        apply_s1(args.dataset, seed)
    summary(args.dataset, seeds)


if __name__ == "__main__":
    main()
