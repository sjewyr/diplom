from __future__ import annotations
import numpy as np
import librosa

from .config import PreprocessConfig, WindowingConfig


def preprocess_audio(path: str, cfg: PreprocessConfig) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=None, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("Пустой аудиосигнал или ошибка чтения")

    if sr != cfg.target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=cfg.target_sr)
        sr = cfg.target_sr

    peak = float(np.max(np.abs(y)) + 1e-12)
    y = (y / peak).astype(np.float32)
    return y, sr


def audio_duration_sec(y: np.ndarray, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return float(len(y) / sr)


def iter_windows(y: np.ndarray, sr: int, cfg: WindowingConfig):
    win = int(cfg.T * sr)
    hop = int(cfg.S * sr)

    if win <= 0 or hop <= 0:
        raise ValueError("Некорректные параметры окна: T/S")

    n = len(y)
    if n < win:
        pad = np.zeros(win - n, dtype=np.float32)
        y = np.concatenate([y, pad])
        n = len(y)

    i = 0
    w_idx = 1

    while i + win <= n:
        a = i
        b = i + win
        w = y[a:b]
        t_start = a / sr
        t_end = b / sr
        yield w_idx, w, t_start, t_end, sr
        w_idx += 1
        i += hop
