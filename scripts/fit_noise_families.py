import math

import numpy as np
from scipy import stats

from configs import heat
from configs.base import JOHNSON_SU_BASE
from src.datasets import make_datasets
from src.pipeline import RESULTS, write_json

N_MC = 2_000_000
SEED = 0
OUT = RESULTS / "heat" / "noise_families.json"


def training_residuals() -> np.ndarray:
    train = make_datasets(heat.DATA)["train"]
    out = []
    for i in range(len(train)):
        item = train[i]
        out.append(item["fut"][:, 0].numpy() - item["hist"][:, 0].numpy()[-24:])
    return np.stack(out).astype(np.float64).ravel()


def family(base: dict) -> dict:
    s0 = JOHNSON_SU_BASE["scale"]
    return dict(base=base, clamp_a=(base["a"] - 1.9, base["a"] + 2.1), clamp_b=(0.02 * base["b"], 1.63 * base["b"]),
                clamp_loc=(-1.5, 2.0), clamp_scale=(base["scale"] * 0.05 / s0, base["scale"] * 2.0 / s0))


def main():
    z = np.random.default_rng(SEED).standard_normal(N_MC)
    jsu = np.sinh((z - JOHNSON_SU_BASE["a"]) / JOHNSON_SU_BASE["b"])
    target_std = JOHNSON_SU_BASE["scale"] * float(jsu.std())
    scale = target_std / float(z.std())
    gaussian = family(dict(a=0.0, b=1.0, loc=-scale * float(z.mean()), scale=scale))
    r = training_residuals()
    sub = np.random.default_rng(SEED).choice(r, size=min(r.size, 50_000), replace=False)
    df_fit = float(stats.t.fit(sub)[0])
    df = max(3, int(round(df_fit)))
    student_t = dict(family(dict(a=0.0, b=1.0, loc=0.0, scale=target_std / math.sqrt(df / (df - 2)))), student_df=df)
    write_json(OUT, dict(target_std=target_std, df_fit=df_fit, gaussian=gaussian, student_t=student_t))
    print(f"Johnson-SU base std {target_std:.6f}; Student-t degrees of freedom {df_fit:.2f} -> {df}; written to {OUT}")


if __name__ == "__main__":
    main()
