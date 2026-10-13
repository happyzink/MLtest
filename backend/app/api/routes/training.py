from fastapi import APIRouter
from app.models.schemas import TrainRequest, TrainResponse
from app.services.data_service import load_dataframe
from app.services.ml_engine import train_model, ALL_ALGOS

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    df = load_dataframe(req.filename)
    result = train_model(df, req.features, req.target, req.task_type, req.algorithm, req.params)
    return TrainResponse(
        algorithm=req.algorithm,
        metrics=result["metrics"],
        params=result["params"],
        split_ratio=result["split_ratio"],
        train_size=result["train_size"],
        test_size=result["test_size"],
        train_predictions=result.get("train_predictions", []),
        test_predictions=result.get("test_predictions", []),
        train_metrics=result.get("train_metrics", {}),
        test_metrics=result.get("test_metrics", {}),
    )


@router.get("/algorithms")
async def list_algorithms():
    result = {}
    for task_type, algos in ALL_ALGOS.items():
        result[task_type] = {
            name: {
                "defaults": entry["defaults"],
                "needs_scaling": entry.get("needs_scaling", False),
            }
            for name, entry in algos.items()
        }
    return result
