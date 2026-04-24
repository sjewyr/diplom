from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

AggMethod = Literal["mean", "weighted_mean"]


@dataclass(frozen=True)
class PreprocessConfig:
    target_sr: int = 16000


@dataclass(frozen=True)
class WindowingConfig:
    T: float = 2.0
    S: float = 0.5


@dataclass(frozen=True)
class PostprocessConfig:
    smoothing: Literal["median_5", "none"] = "median_5"
    theta: float = 0.60
    merge_gap_sec: float = 0.30
    min_total_spoof_sec: float = 1.0


@dataclass(frozen=True)
class EnsembleConfig:
    method: AggMethod = "mean"
    detector_ids: List[str] = field(default_factory=lambda: ["simple_energy", "simple_spectral", "aasist", "rawnet2"])
    weights: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class AppConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    windowing: WindowingConfig = WindowingConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
