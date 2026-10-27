import argparse
import copy

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from configs import DATASETS, dataset_module
from src.baselines import timegrad as tg
from src.datasets import SeriesData, make_datasets, make_loader
from src.pipeline import RESULTS, save_predictions, score_split, write_json
from src.seed import seed_everything

DEVICE = "cuda"
LOADER_WORKERS = 4
WINDOWS_PER_PASS = 256


def cycle(loader):
    while True:
        yield from loader


def validation_loss(net, loader) -> float:
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        for b in loader:
            loss = net(*[v.to(DEVICE) for v in b.values()])[0]
            k = len(next(iter(b.values())))
            total, n = total + loss.item() * k, n + k
    net.train()
    return total / n


def fit(net, train_loader, val_loader) -> list:
    s = tg.TRAINER
    optimizer = Adam(net.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    scheduler = OneCycleLR(optimizer, max_lr=s["maximum_learning_rate"], steps_per_epoch=s["num_batches_per_epoch"],
                           epochs=s["epochs"])
    batches = cycle(train_loader)
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    net.train()
    for epoch in range(1, s["epochs"] + 1):
        total = 0.0
        for _ in range(s["num_batches_per_epoch"]):
            optimizer.zero_grad()
            loss = net(*[v.to(DEVICE) for v in next(batches).values()])[0]
            total += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        val = validation_loss(net, val_loader)
        history.append(dict(epoch=epoch, train_loss=total / s["num_batches_per_epoch"], val_loss=val))
        if val < best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(net.state_dict())
        if epoch - best_epoch >= tg.PATIENCE:
            break
    net.load_state_dict(best_state)
    return history


def run(dataset: str, seed: int) -> dict:
    ds_mod = dataset_module(dataset)
    out = RESULTS / dataset / "timegrad" / f"seed_{seed}"
    seed_everything(seed, deterministic=False)
    series = SeriesData(ds_mod.DATA)
    data = make_datasets(ds_mod.DATA, series)
    train, test = tg.TimeGradWindows(data["train"], training=True), tg.TimeGradWindows(data["test"], training=False)
    network = tg.timegrad_network()
    kwargs = tg.network_kwargs(train)
    net = network.TimeGradTrainingNetwork(**copy.deepcopy(kwargs)).to(DEVICE)
    train_loader = make_loader(train, tg.TRAINER["batch_size"], shuffle=True, seed=seed, drop_last=True,
                               num_workers=LOADER_WORKERS)
    val_loader = make_loader(tg.TimeGradWindows(data["val"], training=True), 256, shuffle=False, seed=seed)
    history = fit(net, train_loader, val_loader)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model.pt")
    pnet = network.TimeGradPredictionNetwork(num_parallel_samples=tg.NUM_SAMPLES, **copy.deepcopy(kwargs)).to(DEVICE)
    pnet.load_state_dict(net.state_dict())
    seed_everything(seed, deterministic=False)
    pred = tg.predict(pnet, test, WINDOWS_PER_PASS, DEVICE)
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
        print(f"TimeGrad {args.dataset} seed {seed}: R2 {m['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
