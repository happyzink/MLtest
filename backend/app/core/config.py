import os
from pathlib import Path
from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"

if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=False)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1500"))

# Aliases for backward compatibility
if not LLM_API_KEY:
    if LLM_PROVIDER in ("openai", "openai_compatible"):
        LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    elif LLM_PROVIDER == "anthropic":
        LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    if not LLM_API_KEY:
        LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

if not LLM_MODEL:
    if LLM_PROVIDER in ("openai", "openai_compatible"):
        LLM_MODEL = "gpt-4o"
    elif LLM_PROVIDER == "anthropic":
        LLM_MODEL = "claude-sonnet-4-20250514"
    elif LLM_PROVIDER == "deepseek":
        LLM_MODEL = "deepseek-chat"

# Tuning budget
TUNING_MAX_TRIALS = int(os.getenv("TUNING_MAX_TRIALS", "10"))
TUNING_PATIENCE = int(os.getenv("TUNING_PATIENCE", "4"))
TUNING_MAX_LLM_CALLS = int(os.getenv("TUNING_MAX_LLM_CALLS", "15"))
TUNING_WALL_TIME_SECONDS = int(os.getenv("TUNING_WALL_TIME_SECONDS", "900"))
TRAIN_TRIAL_TIMEOUT_SECONDS = int(os.getenv("TRAIN_TRIAL_TIMEOUT_SECONDS", "180"))
TRAIN_N_JOBS = int(os.getenv("TRAIN_N_JOBS", "2"))

# Legacy compatibility
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.90"))
MAX_TUNING_ITERATIONS = int(os.getenv("MAX_TUNING_ITERATIONS", "5"))

# Storage
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(_BACKEND_DIR / "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

APP_STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", str(_BACKEND_DIR / "storage")))
APP_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def is_llm_configured() -> bool:
    return bool(LLM_API_KEY and LLM_BASE_URL and LLM_MODEL)


def get_safe_llm_config() -> dict:
    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL if LLM_MODEL else None,
        "base_url": LLM_BASE_URL if LLM_BASE_URL else None,
        "configured": is_llm_configured(),
    }
