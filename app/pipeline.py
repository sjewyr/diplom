from __future__ import annotations
from typing import Dict, Any, List

from .config import AppConfig
from .audio_utils import preprocess_audio, audio_duration_sec, iter_windows
from .ensemble_core import run_ensemble_on_windows, summarize_record_score
from .postproccess import build_intervals, verdict_from_intervals

from .detectors.base import Detector
from .detectors.simple_energy_detector import SimpleEnergyDetector
from .detectors.simple_spectral_detector import SimpleSpectralDetector
from .detectors.aasist_adapter import AASISTDetector
from .detectors.rawnet2_adapter import RawNet2Detector


def _build_detectors(cfg: AppConfig) -> List[Detector]:
    dets: List[Detector] = []

    for detector_id in cfg.ensemble.detector_ids:
        if detector_id == "simple_energy":
            dets.append(SimpleEnergyDetector())
        elif detector_id == "simple_spectral":
            dets.append(SimpleSpectralDetector())
        elif detector_id == "aasist":
            dets.append(AASISTDetector())
        elif detector_id == "rawnet2":
            dets.append(RawNet2Detector())
        else:
            raise ValueError(f"Неизвестный детектор: {detector_id}")

    if not dets:
        raise ValueError("Не выбран ни один детектор")

    return dets


def analyze_file(path: str, original_filename: str, cfg: AppConfig) -> Dict[str, Any]:
    y, sr = preprocess_audio(path, cfg.preprocess)
    duration = audio_duration_sec(y, sr)

    detectors = _build_detectors(cfg)
    for d in detectors:
        d.load()

    windows_iter = iter_windows(y, sr, cfg.windowing)
    ws = run_ensemble_on_windows(detectors, windows_iter, cfg.ensemble)

    intervals = build_intervals(ws, cfg.postprocess)
    verdict = verdict_from_intervals(intervals, cfg.postprocess)
    record_score = summarize_record_score(ws)

    model_rec = {}
    if ws:
        keys = ws[0].model_scores.keys()
        for k in keys:
            model_rec[k] = sum(x.model_scores.get(k, 0.0) for x in ws) / len(ws)

    return {
        "input_filename": original_filename,
        "verdict": verdict,
        "intervals": [
            {"t_start": it.t_start, "t_end": it.t_end, "score": it.score}
            for it in intervals
        ],
        "params": {
            "fs": sr,
            "T": cfg.windowing.T,
            "S": cfg.windowing.S,
            "theta": cfg.postprocess.theta,
            "merge_gap_sec": cfg.postprocess.merge_gap_sec,
            "smoothing": cfg.postprocess.smoothing,
            "ensemble": {
                "method": cfg.ensemble.method,
                "models": cfg.ensemble.detector_ids
            },
        },
        "scores": {
            "record_score": record_score
        },
        "n_windows": len(ws),
        "duration_sec": duration,
        "model_scores": model_rec,
    }


def pipeline_analyze(path: str) -> Dict[str, Any]:
    cfg = AppConfig()
    return analyze_file(path, original_filename=path, cfg=cfg)
