from __future__ import annotations
import numpy as np
import librosa
from .base import Detector, DetectorMeta


class SimpleSpectralDetector(Detector):
    meta = DetectorMeta(
        detector_id="simple_spectral",
        version="1.0",
        input_type="spectrogram"
    )

    def load(self) -> None:
        pass

    def predict_window(self, y: np.ndarray, sr: int) -> float:
        if len(y) < 512:
            return 0.0

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        c = float(np.mean(centroid)) / max(sr / 2, 1.0)
        b = float(np.mean(bandwidth)) / max(sr / 2, 1.0)
        r = float(np.mean(rolloff)) / max(sr / 2, 1.0)

        score = 0.4 * c + 0.3 * b + 0.3 * r
        return float(max(0.0, min(1.0, score)))
