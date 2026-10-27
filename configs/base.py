from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

JOHNSON_SU_BASE = dict(a=0.8984649070073606, b=2.455573716974497, loc=-0.7743342313569153, scale=0.05940356924611432)
HOURLY_FACTORS = (1.2, 1.1, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                  1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0)


@dataclass
class DataConfig:
    name: str
    path: str
    split: tuple
    history: int
    horizon: int = 24
    split_gap: int = 24
    spike_k: float = 1.5
    feature_groups: tuple = ("cal", "tgt")
    absolute_level: bool = False
    normalization: str = "window"
    future_known: tuple = ()


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    diff_inject_every: int = 2
    future_calendar: bool = True
    future_lags: bool = False
    x0_residual: str = "none"


@dataclass
class NoiseConfig:
    family: str = "johnson_su"
    student_df: int = 5
    base: dict = field(default_factory=lambda: dict(JOHNSON_SU_BASE))
    clamp_a: tuple = (-1.0, 3.0)
    clamp_b: tuple = (0.05, 4.0)
    clamp_loc: tuple = (-1.5, 2.0)
    clamp_scale: tuple = (0.05, 2.0)
    hourly_factors: tuple = HOURLY_FACTORS
    refiner_hidden: tuple = (64, 32)
    refiner_dropout: float = 0.1
    spike_modulation: bool = False
    spike_weights_init: tuple = (0.5, 4.0, -0.5, -1.0)
    mixture: str = "none"
    laplace_scale: float = 1.0


@dataclass
class DiffusionConfig:
    lambda_eps: float = 0.6150885307436442
    ddim_steps: int = 5
    timesteps: int = 1200
    beta_start: float = 0.005
    beta_end: float = 0.2
    x0_spike_scale: float = 0.0


@dataclass
class DetectorConfig:
    d_model: int = 256
    dropout: float = 0.1
    n_blocks: int = 2
    d_state: int = 16
    d_conv: int = 4
    alpha: float = 0.9
    head: str = "lead_mlp"
    head_hidden: int = 128


@dataclass
class TrainConfig:
    patience: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    device: str = "cuda"
    num_workers: int = 4
    select_on: str = "val_loss"
    select_windows: int = 0
    amp: bool = False
    compile: bool = False
    window_weight_power: float = 0.0


@dataclass
class ExperimentConfig:
    name: str
    model_type: str
    data: DataConfig
    train: TrainConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: Optional[NoiseConfig] = None
    diffusion: Optional[DiffusionConfig] = None
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    @property
    def needs_spike_probs(self) -> bool:
        return self.noise is not None and (self.noise.spike_modulation or self.noise.mixture != "none")
