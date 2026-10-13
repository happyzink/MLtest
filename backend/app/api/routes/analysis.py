from fastapi import APIRouter
from pydantic import BaseModel
from app.services.data_service import load_dataframe
from app.services.eda_service import compute_correlations, compute_feature_importance, compute_mutual_info

router = APIRouter()


class EDARequest(BaseModel):
    filename: str
    features: list[str]
    target: str
    task_type: str


@router.post("/eda")
async def run_eda(req: EDARequest):
    """Run exploratory data analysis: correlations, feature importance, mutual info."""
    df = load_dataframe(req.filename)

    correlations = compute_correlations(df, req.features)
    feat_imp = compute_feature_importance(df, req.features, req.target, req.task_type)
    mi = compute_mutual_info(df, req.features, req.target, req.task_type)

    return {
        "pearson_corr": correlations["pearson"],
        "spearman_corr": correlations["spearman"],
        "feature_importance": feat_imp,
        "mutual_info": mi,
    }
