import copy
import json
from pathlib import Path

import numpy as np
import torch

from configs.base import ROOT, ExperimentConfig
from src.datasets import SeriesData, WindowDataset, make_datasets
from src.eval.detector_metrics import best_f1_threshold, classification_metrics, sigmoid
from src.eval.evaluate import predict
from src.eval.metrics import scores
from src.models import build_diffusion
from src.models.spike_probs import SpikeProbTable
from src.seed import seed_everything
from src.train.train_baseline import TransformerTrainer
from src.train.train_detector import DetectorTrainer
from src.train.train_diffusion import DiffusionTrainer

RESULTS = ROOT / "results"
PRED_KEYS = ("pred_norm", "y_true", "spike", "mean", "std", "window", "hist_end")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def save_predictions(path: Path, pred: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: pred[k] for k in PRED_KEYS})


def load_predictions(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def detector_dir(dataset: str) -> Path:
    return RESULTS / dataset / "detector"


def probabilities_path(dataset: str) -> Path:
    return detector_dir(dataset) / "probabilities.npz"


def run_detector(cfg: ExperimentConfig, seed: int, out: Path, probability_windows: str = "splits") -> dict:
    seed_everything(seed)
    series = SeriesData(cfg.data)
    data = make_datasets(cfg.data, series)
    if probability_windows == "all":
        data["all"] = WindowDataset(series, np.arange(len(series.windows)), cfg.data)
    trainer = DetectorTrainer(cfg, data, seed)
    history = trainer.fit()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.best_state, out / "detector.pt")
    val, test = trainer.predict("val"), trainer.predict("test")
    threshold = best_f1_threshold(sigmoid(val["logits"]), val["labels"])
    metrics = {split: classification_metrics(p["logits"], p["labels"], threshold, trainer.prevalence_train)
               for split, p in (("val", val), ("test", test))}
    write_json(out / "metrics.json", dict(seed=seed, spike_k=cfg.data.spike_k, metrics=metrics, history=history))
    splits = ("all",) if probability_windows == "all" else ("train", "val", "test")
    np.savez_compressed(out / "probabilities.npz", logits=trainer.all_logits(len(series.windows), splits),
                        hist_end_time=series.index[series.windows["hist_end"].to_numpy()].asi8)
    return metrics


def run_forecaster(cfg: ExperimentConfig, seed: int, out: Path, probs_path: Path = None, eq32: float = 0.0,
                   splits=("val", "test")) -> dict:
    seed_everything(seed)
    series = SeriesData(cfg.data)
    data = make_datasets(cfg.data, series)
    probs = SpikeProbTable(probs_path, series) if cfg.needs_spike_probs else None
    out.mkdir(parents=True, exist_ok=True)
    if cfg.model_type == "diffusion":
        trainer = DiffusionTrainer(cfg, data, seed, probs)
        history = trainer.fit()
        torch.save(trainer.best_states["model"], out / "model.pt")
        torch.save(trainer.best_states["noise"], out / "noise.pt")
        model, noise = trainer.eval_models()
        cfg = copy.deepcopy(cfg)
        cfg.diffusion.x0_spike_scale = eq32
    else:
        trainer = TransformerTrainer(cfg, data, seed)
        history = trainer.fit()
        torch.save(trainer.best_state, out / "model.pt")
        model, noise = trainer.eval_model(), None
    write_json(out / "history.json", history)
    preds = {}
    for split in splits:
        preds[split] = predict(cfg, data[split], model, noise, probs, seed=seed)
        save_predictions(out / f"predictions_{split}.npz", preds[split])
    return preds


def load_forecaster(cfg: ExperimentConfig, out: Path, series: SeriesData, device: str = None):
    model, noise = build_diffusion(cfg, series)
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
    noise.load_state_dict(torch.load(out / "noise.pt", map_location="cpu"))
    device = device or cfg.train.device
    return model.to(device).eval(), noise.to(device).eval()


def score_split(dataset_module, pred: dict, shift=None) -> dict:
    return scores(pred, shift=shift, clip=dataset_module.CLIP_AT_ZERO)
