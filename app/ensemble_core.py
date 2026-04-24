from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from .config import EnsembleConfig
from .detectors.base import Detector


@dataclass
class WindowScore:
    i: int
    t_start: float
    t_end: float
    model_scores: Dict[str, float]
    ensemble_score: float


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def aggregate_window_scores(model_scores: Dict[str, float], cfg: EnsembleConfig) -> float:
    vals = [float(v) for v in model_scores.values()]
    if not vals:
        raise ValueError("Нет оценок моделей")

    if cfg.method == "mean":
        return _clip01(float(np.mean(vals)))

    if cfg.method == "weighted_mean":
        if not cfg.weights:
            raise ValueError("Для weighted_mean нужны weights")
        s = 0.0
        wsum = 0.0
        for k, v in model_scores.items():
            w = float(cfg.weights.get(k, 0.0))
            s += w * float(v)
            wsum += w
        return _clip01(float(s / (wsum + 1e-12)))

    raise ValueError(f"Неизвестный метод агрегации: {cfg.method}")


def run_ensemble_on_windows(
    detectors: List[Detector],
    windows_iter,
    ens_cfg: EnsembleConfig
) -> List[WindowScore]:
    out: List[WindowScore] = []

    for w_idx, w, t_start, t_end, sr in windows_iter:
        scores: Dict[str, float] = {}

        for d in detectors:
            s = float(d.predict_window(w, sr))
            scores[d.meta.detector_id] = _clip01(s)

        ens = aggregate_window_scores(scores, ens_cfg)

        out.append(WindowScore(
            i=w_idx,
            t_start=t_start,
            t_end=t_end,
            model_scores=scores,
            ensemble_score=ens
        ))

    return out


def summarize_record_score(window_scores: List[WindowScore]) -> float:
    if not window_scores:
        return 0.0

    vals = np.array([ws.ensemble_score for ws in window_scores], dtype=np.float32)
    return float(np.quantile(vals, 0.90))
