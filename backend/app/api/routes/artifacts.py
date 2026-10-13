import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import TrainRequest
from app.services.data_service import load_dataframe
from app.services.ml_engine import train_model, _get_algo_class, _validate_params, _prepare_data, ALL_ALGOS
from app.services.artifact_service import (
    RunArtifact, save_pipeline, save_metadata, save_predictions,
    save_metrics, save_effective_params, list_runs, load_metadata,
    _run_dir, create_run_package,
)
from app.services.plot_service import create_scatter_plots
from app.core.config import APP_STORAGE_DIR

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

router = APIRouter()


@router.post("/save")
async def save_model_run(req: TrainRequest):
    """Train, save model artifact, and generate plots."""
    df = load_dataframe(req.filename)
    result = train_model(df, req.features, req.target, req.task_type, req.algorithm, req.params)

    experiment_id = str(uuid.uuid4())[:8]
    run_id = str(uuid.uuid4())[:8]
    run_dir = _run_dir(experiment_id, run_id)

    # Build and save Pipeline
    algo_cls = _get_algo_class(req.algorithm, req.task_type)
    algo_entry = ALL_ALGOS[req.task_type][req.algorithm]
    merged = {**algo_entry["defaults"], **(req.params or {})}
    valid_params = _validate_params(algo_cls, merged)

    X_train, X_test, y_train, y_test, scaler, split_desc, target_le = _prepare_data(
        df, req.features, req.target, req.task_type
    )

    steps = []
    if algo_entry.get("needs_scaling", False):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", algo_cls(**valid_params)))
    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)
    save_pipeline(pipeline, run_dir)

    # Save predictions
    save_predictions(result.get("train_predictions", []), "train", run_dir)
    save_predictions(result.get("test_predictions", []), "test", run_dir)

    # Save metadata
    artifact = RunArtifact(
        run_id=run_id,
        experiment_id=experiment_id,
        algorithm=req.algorithm,
        task_type=req.task_type,
        target=req.target,
        features=req.features,
        effective_params=result["params"],
        train_metrics=result.get("train_metrics", {}),
        test_metrics=result.get("test_metrics", {}),
        split_info={"split_ratio": result["split_ratio"], "train_size": result["train_size"], "test_size": result["test_size"]},
    )
    save_metadata(artifact, run_dir)
    save_effective_params(result["params"], run_dir)
    save_metrics(
        result.get("train_metrics", {}),
        result.get("test_metrics", {}),
        run_dir=run_dir,
    )

    # Generate plots for regression
    if req.task_type == "regression" and result.get("train_predictions"):
        train_arr = result["train_predictions"]
        test_arr = result["test_predictions"]
        if train_arr and test_arr:
            create_scatter_plots(
                train_y_true=np.array([p["y_true"] for p in train_arr]),
                train_y_pred=np.array([p["y_pred"] for p in train_arr]),
                test_y_true=np.array([p["y_true"] for p in test_arr]),
                test_y_pred=np.array([p["y_pred"] for p in test_arr]),
                target_name=req.target,
                output_dir=run_dir / "figures",
            )

    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "algorithm": req.algorithm,
        "metrics": result["metrics"],
        "params": result["params"],
        "train_predictions": result.get("train_predictions", []),
        "test_predictions": result.get("test_predictions", []),
        "train_metrics": result.get("train_metrics", {}),
        "test_metrics": result.get("test_metrics", {}),
        "split_ratio": result["split_ratio"],
        "train_size": result["train_size"],
        "test_size": result["test_size"],
    }


@router.get("/runs")
async def get_runs(experiment_id: str = None):
    return list_runs(experiment_id)


@router.get("/runs/{experiment_id}/{run_id}")
async def get_run(experiment_id: str, run_id: str):
    run_dir = _run_dir(experiment_id, run_id)
    meta = load_metadata(run_dir)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")
    return meta


@router.get("/runs/{experiment_id}/{run_id}/download")
async def download_model(experiment_id: str, run_id: str):
    output_dir = APP_STORAGE_DIR / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = create_run_package(experiment_id, run_id, output_dir)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Package not found")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"model_{run_id}.zip",
    )
