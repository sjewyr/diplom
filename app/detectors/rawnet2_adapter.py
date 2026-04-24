from __future__ import annotations
import os
import numpy as np
import torch

from .base import Detector, DetectorMeta


class RawNet2Detector(Detector):
    meta = DetectorMeta(
        detector_id="rawnet2",
        version="1.0",
        input_type="waveform"
    )

    def __init__(self, weights_path: str = "weights/rawnet2.pt", device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Не найдены веса RawNet2: {self.weights_path}"
            )

        try:
            from external_models.rawnet2 import RawNet2  # type: ignore
        except Exception as e:
            raise ImportError(
                "Не найден external_models.rawnet2.RawNet2"
            ) from e

        self.model = RawNet2().to(self.device)
        state = torch.load(self.weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def predict_window(self, y: np.ndarray, sr: int) -> float:
        if self.model is None:
            raise RuntimeError("RawNet2 не загружен")

        x = torch.from_numpy(y.astype(np.float32))[None, :]
        x = x.to(self.device)

        with torch.inference_mode():
            out = self.model(x)

        s = float(out.squeeze().detach().cpu().numpy())
        p = 1.0 / (1.0 + np.exp(-s))
        return float(max(0.0, min(1.0, p)))
