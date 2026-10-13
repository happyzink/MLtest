from fastapi import APIRouter
from app.models.schemas import TuneRequest, TuneResponse
from app.services.data_service import load_dataframe
from app.services.llm_tuner import run_tuning_loop

router = APIRouter()


@router.post("/auto-tune", response_model=TuneResponse)
async def auto_tune(req: TuneRequest):
    df = load_dataframe(req.filename)

    result = await run_tuning_loop(
        df=df,
        features=req.features,
        target=req.target,
        task_type=req.task_type,
        algorithm=req.algorithm,
    )

    return TuneResponse(**result)
