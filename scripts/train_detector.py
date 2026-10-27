import argparse

from configs import DATASETS, dataset_module
from src.pipeline import detector_dir, run_detector

COLUMNS = ("spike_rate", "auprc", "auroc", "f1", "precision", "recall", "brier_skill", "ece", "threshold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    args = parser.parse_args()
    ds = dataset_module(args.dataset)
    metrics = run_detector(ds.detector(), ds.DETECTOR_SEED, detector_dir(args.dataset), ds.PROBABILITY_WINDOWS)
    print("| split | " + " | ".join(COLUMNS) + " |")
    for split, m in metrics.items():
        print(f"| {split} | " + " | ".join(f"{m[c]:.3f}" for c in COLUMNS) + " |")


if __name__ == "__main__":
    main()
