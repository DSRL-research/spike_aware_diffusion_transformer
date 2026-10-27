from configs.base import ExperimentConfig
from src.models.baseline_transformer import TransformerDecoderAR
from src.models.diffusion_transformer import DiffusionTransformer
from src.models.noise import NoiseSampler
from src.models.spike_detector import SpikeDetector


def build_diffusion(cfg: ExperimentConfig, series) -> tuple:
    model = DiffusionTransformer(cfg.model, cfg.data.history, cfg.data.horizon, cfg.diffusion.timesteps,
                                 len(series.hist_feature_names), len(series.fut_feature_names))
    noise = NoiseSampler(cfg.noise, cfg.diffusion, series.context_dim)
    return model, noise


def build_transformer(cfg: ExperimentConfig, series) -> TransformerDecoderAR:
    return TransformerDecoderAR(cfg.model, cfg.data.history, cfg.data.horizon, len(series.hist_feature_names),
                                len(series.fut_feature_names))


def build_detector(cfg: ExperimentConfig, series) -> SpikeDetector:
    return SpikeDetector(cfg.detector, cfg.data.horizon, cfg.data.history, len(series.hist_feature_names),
                         len(series.fut_feature_names))
