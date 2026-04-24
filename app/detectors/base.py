from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class DetectorMeta:
    detector_id: str
    version: str
    input_type: str


class Detector:
    meta: DetectorMeta

    def load(self) -> None:
        raise NotImplementedError

    def predict_window(self, y: np.ndarray, sr: int) -> float:
        raise NotImplementedError

    def extra_info(self) -> Dict[str, Any]:
        return {}
