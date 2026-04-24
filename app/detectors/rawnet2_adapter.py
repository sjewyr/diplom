from __future__ import annotations
import os
import numpy as np
import torch

from .base import Detector, DetectorMeta

# Default RawNet architecture config (matches external_models/rawnet/model_config_RawNet.yaml)
_DEFAULT_RAWNET_ARGS = {
    "nb_samp": 64600,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
}


def _pad_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    """Повторяет сигнал до max_len или обрезает — как в оригинальном data_utils.pad."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = (max_len // x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class RawNet2Detector(Detector):
    meta = DetectorMeta(
        detector_id="rawnet2",
        version="1.0",
        input_type="waveform"
    )

    def __init__(
        self,
        weights_path: str = "external_models/rawnet/checkpoints/best_model.pth",
        config_path: str = "external_models/rawnet/model_config_RawNet.yaml",
        device: str = "cpu",
        d_args: dict | None = None,
    ):
        self.weights_path = weights_path
        self.config_path = config_path
        self.device = device
        self.d_args = d_args
        self.model = None
        self.nb_samp = _DEFAULT_RAWNET_ARGS["nb_samp"]

    def _resolve_d_args(self) -> dict:
        if self.d_args is not None:
            return dict(self.d_args)

        if self.config_path and os.path.exists(self.config_path):
            try:
                import yaml  # type: ignore
                with open(self.config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict) and "model" in cfg:
                    return dict(cfg["model"])
            except ImportError:
                # pyyaml не установлен — используем встроенный дефолт
                pass

        return dict(_DEFAULT_RAWNET_ARGS)

    def load(self) -> None:
        if self.model is not None:
            return

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Не найдены веса RawNet: {self.weights_path}"
            )

        from external_models.rawnet import RawNet  # класс модели

        d_args = self._resolve_d_args()
        self.nb_samp = int(d_args.get("nb_samp", self.nb_samp))

        self.model = RawNet(d_args, self.device).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def predict_window(self, y: np.ndarray, sr: int) -> float:
        if self.model is None:
            raise RuntimeError("RawNet не загружен — вызовите load()")

        # Модель ожидает nb_samp (64600) сэмплов — паддим/тримим
        y_fixed = _pad_or_trim(y.astype(np.float32), self.nb_samp)

        # forward ожидает (batch, seq_len)
        x = torch.from_numpy(y_fixed).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(x)  # log_softmax (batch, 2): [0]=fake/spoof, [1]=real

        # Модель уже выдаёт log_softmax → преобразуем в вероятности
        probs = torch.exp(output)
        spoof_prob = float(probs[0, 0].detach().cpu().item())
        return float(max(0.0, min(1.0, spoof_prob)))
