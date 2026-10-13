"""
Artifact service: save/load models, predictions, and metadata.
Uses joblib for Pipeline persistence and JSON for metadata.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import APP_STORAGE_DIR


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class RunArtifact:
    run_id: str
    experiment_id: str
    algorithm: str
    task_type: str
    target: str
    features: list[str]
    effective_params: dict
    train_metrics: dict
    test_metrics: dict
    cv_metrics: Optional[dict] = None
    split_info: Optional[dict] = None
    created_at: str = ""
    status: str = "succeeded"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


def _run_dir(experiment_id: str, run_id: str) -> Path:
    d = APP_STORAGE_DIR / "experiments" / experiment_id / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_pipeline(pipeline, run_dir: Path):
    path = run_dir / "pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


def load_pipeline(run_dir: Path):
    path = run_dir / "pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found: {path}")
    return joblib.load(path)


def save_predictions(predictions: list[dict], split: str, run_dir: Path):
    if not predictions:
        return
    df = pd.DataFrame(predictions)
    df.to_csv(run_dir / f"{split}_predictions.csv", index=False)


def load_predictions(split: str, run_dir: Path) -> pd.DataFrame:
    path = run_dir / f"{split}_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_metadata(artifact: RunArtifact, run_dir: Path):
    path = run_dir / "metadata.json"
    data = asdict(artifact)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyEncoder), encoding="utf-8")


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_effective_params(params: dict, run_dir: Path):
    path = run_dir / "effective_params.json"
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False, cls=NumpyEncoder), encoding="utf-8")


def save_metrics(train_metrics: dict, test_metrics: dict, cv_metrics: dict = None, run_dir: Path = None):
    data = {
        "train": train_metrics,
        "test": test_metrics,
        "cv": cv_metrics,
    }
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyEncoder), encoding="utf-8")


def list_runs(experiment_id: str = None) -> list[dict]:
    results = []
    base = APP_STORAGE_DIR / "experiments"
    if not base.exists():
        return results

    exp_dirs = [base / experiment_id] if experiment_id else base.iterdir()
    for exp_dir in exp_dirs:
        runs_dir = exp_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                meta = load_metadata(run_dir)
                if meta:
                    results.append(meta)
    return results


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_run_package(experiment_id: str, run_id: str, output_path: Path) -> Path:
    """Create a ZIP package of a run for download."""
    run_dir = _run_dir(experiment_id, run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {experiment_id}/{run_id}")

    zip_path = output_path / f"model_{run_id}"
    shutil.make_archive(str(zip_path), "zip", str(run_dir))
    return zip_path.with_suffix(".zip")
