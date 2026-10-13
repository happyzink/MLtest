from pydantic import BaseModel, Field
from typing import Optional, Any


class UploadResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]
    preview: list[dict[str, Any]]


class FeatureSelection(BaseModel):
    filename: str
    features: list[str]
    target: str
    task_type: str


class EDAResponse(BaseModel):
    pearson_corr: dict[str, dict[str, float]]
    spearman_corr: dict[str, dict[str, float]]
    feature_importance: dict[str, float]
    mutual_info: dict[str, float]


class TrainRequest(BaseModel):
    filename: str
    features: list[str]
    target: str
    task_type: str
    algorithm: str
    params: Optional[dict[str, Any]] = None


class PredictionPoint(BaseModel):
    y_true: float
    y_pred: float


class TrainResponse(BaseModel):
    algorithm: str
    metrics: dict[str, Any]
    params: dict[str, Any]
    split_ratio: str
    train_size: int
    test_size: int
    train_predictions: list[PredictionPoint] = []
    test_predictions: list[PredictionPoint] = []
    train_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}


class TuneRequest(BaseModel):
    filename: str
    features: list[str]
    target: str
    task_type: str
    algorithm: str
    # No API key from frontend - backend reads from .env


class TuneIteration(BaseModel):
    round: int
    params: dict[str, Any]
    metrics: dict[str, Any]
    score: Optional[float] = None
    rationale: Optional[str] = None
    status: str = "succeeded"
    error: Optional[str] = None


class TuneResponse(BaseModel):
    iterations: list[TuneIteration]
    best_params: dict[str, Any]
    best_score: Optional[float] = None
    reached_threshold: bool
    final_report: str
    stop_reason: Optional[str] = None
    llm_calls: int = 0


class LLMStatusResponse(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    configured: bool = False
    connection_status: Optional[str] = None
