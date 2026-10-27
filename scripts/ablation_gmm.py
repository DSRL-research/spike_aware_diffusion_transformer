import copy

from configs import heat
from src.datasets import SeriesData, make_datasets
from src.eval.evaluate import predict
from src.eval.metrics import KEYS, format_row, table_header
from src.integration import GMMSpikeProbs, s1_shift
from src.models.spike_probs import SpikeProbTable
from src.pipeline import RESULTS, load_forecaster, load_predictions, probabilities_path, score_split, write_json
from src.seed import seed_everything

NAMES = {"diffusion": "Diffusion only", "s1": "S1 (detector)", "s1_gmm": "S1 (GMM)", "s2": "S2 (detector)",
         "s2_gmm": "S2 (GMM)", "s3": "S3 (detector)", "s3_gmm": "S3 (GMM)"}


def main():
    unit, scale = heat.UNIT
    series = SeriesData(heat.DATA)
    data = make_datasets(heat.DATA, series)
    detector = SpikeProbTable(probabilities_path("heat"), series)
    kappa, beta = heat.S1
    rows = {k: [] for k in NAMES}
    for seed in heat.SEEDS:
        gmm = GMMSpikeProbs(detector.probs, data["val"].indices, seed=seed)
        pred = load_predictions(RESULTS / "heat" / "diffusion" / f"seed_{seed}" / "predictions_test.npz")
        rows["diffusion"].append(score_split(heat, pred))
        rows["s1"].append(score_split(heat, pred, s1_shift(detector.probs[pred["window"]], kappa, beta)))
        rows["s1_gmm"].append(score_split(heat, pred, s1_shift(gmm.table(pred["window"]), kappa, beta)))
        for v in ("s2", "s3"):
            cfg = copy.deepcopy(heat.forecaster(v))
            cfg.diffusion.x0_spike_scale = heat.EQ32[v]
            model, noise = load_forecaster(cfg, RESULTS / "heat" / v / f"seed_{seed}", series)
            for name, probs in ((v, detector), (f"{v}_gmm", gmm)):
                seed_everything(seed)
                rows[name].append(score_split(heat, predict(cfg, data["test"], model, noise, probs, seed=seed)))
        print(f"seed {seed}: " + ", ".join(f"{k} R2 {v[-1]['r2']:.4f}" for k, v in rows.items()), flush=True)
    scaled = {k: [{m: x[m] * (1 if m in ("r2", "smape") else scale) for m in KEYS} for x in v] for k, v in rows.items()}
    seeds = ", ".join(map(str, heat.SEEDS))
    lines = [f"# Heat demand: detector vs GMM spike probabilities, test, seeds {seeds} ({unit})", ""]
    lines += table_header() + [format_row(NAMES[k], v) for k, v in scaled.items()]
    write_json(RESULTS / "heat" / "gmm.json", scaled)
    (RESULTS / "heat" / "gmm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
