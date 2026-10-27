import argparse
import copy

import numpy as np
import torch

from configs import DATASETS, dataset_module
from src.baselines import dlinear as dl
from src.datasets import SeriesData, make_datasets, make_loader
from src.pipeline import RESULTS, save_predictions, score_split, write_json
from src.seed import seed_everything

DEVICE = "cuda"
LOADER_WORKERS = 4


def fit(model, train_loader, val_loader) -> list:
    s = dl.TRAINING
    optimizer = torch.optim.Adam(model.parameters(), lr=s["learning_rate"])
    criterion = torch.nn.MSELoss()
    best, best_state, counter, history = None, None, 0, []
    for epoch in range(s["train_epochs"]):
        model.train()
        losses = []
        for b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b["x"].to(DEVICE)), b["y"].to(DEVICE))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(np.average([criterion(model(b["x"].to(DEVICE)), b["y"].to(DEVICE)).item() for b in val_loader]))
        history.append(dict(epoch=epoch + 1, train_loss=float(np.average(losses)), val_loss=val))
        if best is None or val < best:
            best, best_state, counter = val, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
            if counter >= s["patience"]:
                break
        dl.adjust_learning_rate(optimizer, epoch + 1, s["learning_rate"])
    model.load_state_dict(best_state)
    return history


def run(dataset: str, seed: int) -> dict:
    ds_mod = dataset_module(dataset)
    out = RESULTS / dataset / "dlinear" / f"seed_{seed}"
    seed_everything(seed)
    series = SeriesData(ds_mod.DATA)
    data = make_datasets(ds_mod.DATA, series)
    model = dl.build_model(ds_mod.DATA.history, ds_mod.DATA.horizon).to(DEVICE)
    train, val, test = (dl.DLinearWindows(data[k]) for k in ("train", "val", "test"))
    train_loader = make_loader(train, dl.TRAINING["batch_size"], shuffle=True, seed=seed, drop_last=True,
                               num_workers=LOADER_WORKERS)
    history = fit(model, train_loader, make_loader(val, 256, shuffle=False, seed=seed, num_workers=LOADER_WORKERS))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    pred = dl.predict(model, make_loader(test, 256, shuffle=False, seed=seed, num_workers=LOADER_WORKERS), DEVICE)
    save_predictions(out / "predictions_test.npz", pred)
    metrics = score_split(ds_mod, pred)
    write_json(out / "metrics.json", dict(test=metrics, history=history))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    for seed in args.seeds or dataset_module(args.dataset).SEEDS:
        m = run(args.dataset, seed)
        print(f"DLinear {args.dataset} seed {seed}: R2 {m['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
