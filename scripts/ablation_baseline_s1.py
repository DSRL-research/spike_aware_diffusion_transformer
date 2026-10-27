from configs import heat
from src.datasets import SeriesData
from src.eval.metrics import KEYS, format_row, table_header
from src.integration import s1_shift
from src.models.spike_probs import SpikeProbTable
from src.pipeline import RESULTS, load_predictions, probabilities_path, score_split, write_json

NAMES = {"transformer": "Transformer", "transformer_s1": "Transformer + S1", "diffusion": "Diffusion only",
         "diffusion_s1": "Diffusion only + S1"}


def main():
    unit, scale = heat.UNIT
    table = SpikeProbTable(probabilities_path("heat"), SeriesData(heat.DATA)).probs
    kappa, beta = heat.S1
    rows = {k: [] for k in NAMES}
    for seed in heat.SEEDS:
        for model in ("transformer", "diffusion"):
            pred = load_predictions(RESULTS / "heat" / model / f"seed_{seed}" / "predictions_test.npz")
            rows[model].append(score_split(heat, pred))
            rows[f"{model}_s1"].append(score_split(heat, pred, s1_shift(table[pred["window"]], kappa, beta)))
    scaled = {k: [{m: x[m] * (1 if m in ("r2", "smape") else scale) for m in KEYS} for x in v] for k, v in rows.items()}
    seeds = ", ".join(map(str, heat.SEEDS))
    lines = [f"# Heat demand: S1 on the transformer baseline and on diffusion only, test, seeds {seeds} ({unit})", ""]
    lines += table_header() + [format_row(NAMES[k], v) for k, v in scaled.items()]
    write_json(RESULTS / "heat" / "baseline_s1.json", scaled)
    (RESULTS / "heat" / "baseline_s1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
