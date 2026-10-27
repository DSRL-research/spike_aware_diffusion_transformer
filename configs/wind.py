from dataclasses import replace

from configs.base import (DataConfig, DetectorConfig, DiffusionConfig, ExperimentConfig, ModelConfig, NoiseConfig,
                          TrainConfig)

SEEDS = (42, 43, 44, 45, 46)
DETECTOR_SEED = 0
S1 = (0.9, 0.1)
EQ32 = {"s2": 0.0, "s3": 0.0}
CLIP_AT_ZERO = True
UNIT = ("kW", 1.0)
PROBABILITY_WINDOWS = "all"

DATA = DataConfig(name="wind", path="data/processed/wind.parquet", split=(0.6, 0.2, 0.2), history=72,
                  feature_groups=("cal", "obs", "tgt"), normalization="global",
                  future_known=("obs__theoretical_power",))
VANILLA_DATA = replace(DATA, feature_groups=("cal", "obs"))
VANILLA_TRAINING = dict(lr=5e-4, weight_decay=1e-5, batch_size=32, epochs=50, patience=5, grad_clip=1.0)


def detector() -> ExperimentConfig:
    return ExperimentConfig(name="wind/detector", model_type="detector", data=DATA,
                            train=TrainConfig(patience=6, epochs=40, batch_size=64, lr=5e-4),
                            detector=DetectorConfig(d_model=256, head="lead_mlp"))


def forecaster(variant: str = "diffusion", noise: NoiseConfig = None) -> ExperimentConfig:
    noise = replace(noise or NoiseConfig(), spike_modulation=variant == "s2",
                    mixture="laplace" if variant == "s3" else "none")
    return ExperimentConfig(
        name=f"wind/{variant}", model_type="diffusion", data=DATA,
        train=TrainConfig(patience=10, epochs=60, batch_size=32, lr=1e-4, amp=True, compile=True),
        model=ModelConfig(d_model=256, n_layers=4, dropout=0.1, x0_residual="last"),
        noise=noise, diffusion=DiffusionConfig())
