from __future__ import annotations
import numpy as np
from .base import Detector, DetectorMeta


class SimpleEnergyDetector(Detector):
    meta = DetectorMeta(
        detector_id="simple_energy",
        version="1.0",
        input_type="waveform"
    )

    def load(self) -> None:
        pass

    def predict_window(self, y: np.ndarray, sr: int) -> float:
        eps = 1e-12
        rms = float(np.sqrt(np.mean(y ** 2) + eps))
        zcr = float(((y[:-1] * y[1:]) < 0).mean()) if len(y) > 1 else 0.0

        score = 0.5 * min(1.0, zcr * 10.0) + 0.5 * min(1.0, rms * 3.0)
        return float(max(0.0, min(1.0, score)))
