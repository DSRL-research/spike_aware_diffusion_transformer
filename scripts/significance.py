import argparse

from configs import DATASETS, dataset_module
from src.eval.significance import METRICS, paired_tests, window_errors
from src.pipeline import RESULTS, load_predictions, write_json

OURS = ("s1", "s2", "s3")
BASELINES = ("diffusion", "transformer", "dlinear", "timegrad", "csdi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    args = parser.parse_args()
    ds = dataset_module(args.dataset)
    unit, scale = ds.UNIT
    errors, hist_end, has_spike = {}, None, None
    for model in OURS + BASELINES:
        preds = [load_predictions(RESULTS / args.dataset / model / f"seed_{s}" / "predictions_test.npz")
                 for s in ds.SEEDS]
        errors[model], he, hs = window_errors(preds, clip=ds.CLIP_AT_ZERO)
        if hist_end is None:
            hist_end, has_spike = he, hs
    tests = paired_tests(errors, [(a, b) for a in OURS for b in BASELINES], hist_end, has_spike, scale)
    lines = [f"# {args.dataset}: paired Wilcoxon signed-rank tests, per-window errors averaged over seeds ({unit})", "",
             "| strategy | baseline | metric | Δ (all) | r (all) | p Holm (all) | Δ (non-overl.) | r (non-overl.) | "
             "p Holm (non-overl.) |", "|---|---|---|---|---|---|---|---|---|"]
    for a in OURS:
        for b in BASELINES:
            for metric in METRICS:
                ta, tn = tests[("all", a, b, metric)], tests[("non-overlapping", a, b, metric)]
                lines.append(f"| {a} | {b} | {metric} | {ta['delta']:+.2f} | {ta['r']:+.2f} | {ta['p_holm']:.2g} | "
                             f"{tn['delta']:+.2f} | {tn['r']:+.2f} | {tn['p_holm']:.2g} |")
    write_json(RESULTS / args.dataset / "significance.json", {"|".join(k): v for k, v in tests.items()})
    (RESULTS / args.dataset / "significance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
