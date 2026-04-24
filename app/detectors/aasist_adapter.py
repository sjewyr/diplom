from __future__ import annotations
import os
import numpy as np
import torch

from .base import Detector, DetectorMeta

# Default AASIST architecture config (matches the original paper / repo defaults)
_DEFAULT_AASIST_ARGS = {
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0],
    "first_conv": 128,
}


class AASISTDetector(Detector):
    meta = DetectorMeta(
        detector_id="aasist",
        version="1.0",
        input_type="waveform"
    )

    def __init__(
        self,
        weights_path: str = "external_models/aasist/models/weights/AASIST.pth",
        device: str = "cpu",
        d_args: dict | None = None,
    ):
        self.weights_path = weights_path
        self.device = device
        self.d_args = d_args or dict(_DEFAULT_AASIST_ARGS)
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Не найдены веса AASIST: {self.weights_path}"
            )

        from external_models.aasist.models import AASIST  # Model class

        self.model = AASIST(self.d_args).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def predict_window(self, y: np.ndarray, sr: int) -> float:
        if self.model is None:
            raise RuntimeError("AASIST не загружен — вызовите load()")

        # AASIST.forward expects (batch, seq_len) — it does unsqueeze(1) internally
        x = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)  # (1, seq_len)
        x = x.to(self.device)

        with torch.inference_mode():
            _last_hidden, output = self.model(x)  # output shape: (batch, 2)

        # output[:, 1] = spoof logit, output[:, 0] = bonafide logit
        # Apply softmax to get probability of spoof
        probs = torch.softmax(output, dim=1)
        spoof_prob = float(probs[0, 1].detach().cpu().item())
        return float(max(0.0, min(1.0, spoof_prob)))
