from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from .config import PostprocessConfig
from .ensemble_core import WindowScore


@dataclass
class Interval:
    t_start: float
    t_end: float
    score: float


def median_smooth(scores: np.ndarray, k: int = 5) -> np.ndarray:
    if len(scores) == 0:
        return scores
    if k <= 1:
        return scores

    k = int(k)
    pad = k // 2
    x = np.pad(scores, (pad, pad), mode="edge")
    out = np.empty_like(scores)

    for i in range(len(scores)):
        out[i] = float(np.median(x[i:i + k]))
    return out


def _interval_from_range(window_scores: List[WindowScore], smooth: np.ndarray, a: int, b: int) -> Interval:
    t_start = float(window_scores[a].t_start)
    t_end = float(window_scores[b].t_end)
    score = float(np.max(smooth[a:b + 1]))
    return Interval(t_start=t_start, t_end=t_end, score=score)


def build_intervals(window_scores: List[WindowScore], cfg: PostprocessConfig) -> List[Interval]:
    if not window_scores:
        return []

    raw = np.array([ws.ensemble_score for ws in window_scores], dtype=np.float32)

    if cfg.smoothing == "median_5":
        smooth = median_smooth(raw, k=5)
    else:
        smooth = raw

    mask = smooth >= float(cfg.theta)

    intervals: List[Interval] = []
    start_idx = None

    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        if (not flag) and start_idx is not None:
            a = start_idx
            b = i - 1
            intervals.append(_interval_from_range(window_scores, smooth, a, b))
            start_idx = None

    if start_idx is not None:
        intervals.append(_interval_from_range(window_scores, smooth, start_idx, len(mask) - 1))

    merged: List[Interval] = []
    for it in intervals:
        if not merged:
            merged.append(it)
            continue

        prev = merged[-1]
        gap = it.t_start - prev.t_end

        if gap <= float(cfg.merge_gap_sec):
            merged[-1] = Interval(
                t_start=prev.t_start,
                t_end=max(prev.t_end, it.t_end),
                score=max(prev.score, it.score)
            )
        else:
            merged.append(it)

    return merged


def verdict_from_intervals(intervals: List[Interval], cfg: PostprocessConfig) -> str:
    total = 0.0
    for it in intervals:
        total += max(0.0, it.t_end - it.t_start)

    if total >= float(cfg.min_total_spoof_sec):
        return "spoof_detected"
    return "no_spoof_detected"
