from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class IntervalOut(BaseModel):
    t_start: float = Field(..., ge=0.0)
    t_end: float = Field(..., ge=0.0)
    score: float = Field(..., ge=0.0, le=1.0)


class EnsembleParams(BaseModel):
    method: str
    models: List[str]


class ParamsOut(BaseModel):
    fs: int
    T: float
    S: float
    theta: float
    merge_gap_sec: float
    smoothing: str
    ensemble: EnsembleParams


class AnalyzeOut(BaseModel):
    input_filename: str
    verdict: str
    intervals: List[IntervalOut]
    params: ParamsOut
    scores: Dict[str, Any]
    runtime_sec: float
    n_windows: int
    duration_sec: float
    model_scores: Optional[Dict[str, Any]] = None
