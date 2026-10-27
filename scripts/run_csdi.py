import argparse

from configs import DATASETS, dataset_module
from src.baselines.csdi import CSDIWindows, csdi_config, csdi_modules, predict
from src.datasets import SeriesData, make_datasets, make_loader
from src.pipeline import RESULTS, save_predictions, score_split, write_json
from src.seed import seed_everything

DEVICE = "cuda"
N_DRAWS = 5
SETUP = {"heat": dict(known_lags=True, csdi={}, rows=80), "wind": dict(known_lags=False,
                                                                       csdi={"train.itr_per_epoch": 700}, rows=160)}


def run(dataset: str, seed: int) -> dict:
    ds_mod, setup = dataset_module(dataset), SETUP[dataset]
    out = RESULTS / dataset / "csdi" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    series = SeriesData(ds_mod.DATA)
    data = make_datasets(ds_mod.DATA, series)
    train = CSDIWindows(data["train"], known_lags=setup["known_lags"])
    test = CSDIWindows(data["test"], known_lags=setup["known_lags"])
    main_model, utils = csdi_modules()
    cfg = csdi_config(setup["csdi"])
    model = main_model.CSDI_Forecasting(cfg, DEVICE, len(train.channels)).to(DEVICE)
    utils.train(model, cfg["train"], make_loader(train, cfg["train"]["batch_size"], True, seed), valid_loader=None,
                foldername=str(out))
    seed_everything(seed)
    pred = predict(model, test, N_DRAWS, setup["rows"])
    save_predictions(out / "predictions_test.npz", pred)
    metrics = score_split(ds_mod, pred)
    write_json(out / "metrics.json", dict(test=metrics))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    for seed in args.seeds or dataset_module(args.dataset).SEEDS:
        m = run(args.dataset, seed)
        print(f"CSDI {args.dataset} seed {seed}: R2 {m['r2']:.4f}", flush=True)


if __name__ == "__main__":
    main()
