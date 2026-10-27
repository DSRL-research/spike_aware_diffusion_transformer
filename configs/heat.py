from dataclasses import replace

from configs.base import (DataConfig, DetectorConfig, DiffusionConfig, ExperimentConfig, ModelConfig, NoiseConfig,
                          TrainConfig)

SEEDS = (42, 43, 44, 45, 46)
DETECTOR_SEED = 0
S1 = (0.1, 0.75)
EQ32 = {"s2": 0.2, "s3": 0.2}
CLIP_AT_ZERO = False
UNIT = ("kW", 1e-3)
PROBABILITY_WINDOWS = "splits"

DATA = DataConfig(name="heat", path="data/processed/heat.parquet", split=(0.7, 0.2, 0.1), history=168,
                  feature_groups=("cal", "tgt"), absolute_level=True)
DETECTOR_DATA = replace(DATA, feature_groups=("cal", "obs", "tgt"), absolute_level=False)


def detector() -> ExperimentConfig:
    return ExperimentConfig(name="heat/detector", model_type="detector", data=DETECTOR_DATA,
                            train=TrainConfig(patience=6, epochs=40, batch_size=64, lr=5e-4),
                            detector=DetectorConfig(d_model=64, head="same_hour"))


def forecaster(variant: str = "diffusion", noise: NoiseConfig = None) -> ExperimentConfig:
    noise = replace(noise or NoiseConfig(), spike_modulation=variant == "s2",
                    mixture="laplace" if variant == "s3" else "none")
    return ExperimentConfig(
        name=f"heat/{variant}", model_type="diffusion", data=DATA,
        train=TrainConfig(patience=8, epochs=60, batch_size=16, lr=1e-5, select_on="sampled_mae", select_windows=512,
                          amp=True, compile=True, window_weight_power=2.0),
        model=ModelConfig(d_model=128, n_layers=6, dropout=0.3, future_lags=True, x0_residual="same_hour_mean"),
        noise=noise, diffusion=DiffusionConfig())


def transformer() -> ExperimentConfig:
    return ExperimentConfig(name="heat/transformer", model_type="transformer", data=DATA,
                            train=TrainConfig(patience=3, epochs=100, batch_size=16, lr=3e-5),
                            model=ModelConfig(d_model=256, n_layers=6, dropout=0.1, future_lags=True,
                                              x0_residual="same_hour_mean"))
