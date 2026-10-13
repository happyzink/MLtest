from fastapi import APIRouter
from app.models.schemas import LLMStatusResponse
from app.core.config import get_safe_llm_config, is_llm_configured

router = APIRouter()


@router.get("/llm", response_model=LLMStatusResponse)
async def get_llm_status():
    return LLMStatusResponse(**get_safe_llm_config())


@router.post("/llm/test")
async def test_llm_connection():
    """Test LLM connection with a minimal request."""
    config = get_safe_llm_config()
    if not config["configured"]:
        return {"success": False, "error": "LLM 未配置：缺少 API Key、Base URL 或 Model"}

    try:
        from app.services.llm_client import test_connection
        result = await test_connection()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
