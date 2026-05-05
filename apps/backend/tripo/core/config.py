import os
from pathlib import Path
from dotenv import load_dotenv

_BASE = Path(__file__).parent.parent
load_dotenv(_BASE / ".env")

TRIPO_API_KEY: str = os.environ["TRIPO_API_KEY"]
TRIPO_BASE_URL: str = "https://api.tripo3d.ai/v2/openapi"
TRIPO_MODEL_VERSION: str = os.getenv("TRIPO_MODEL_VERSION", "v3.0")

STORAGE_CAPTURES: Path = _BASE / "storage" / "captures"
STORAGE_MODELS: Path = _BASE / "storage" / "models"

# Unity External Assets 경로 (절대경로 또는 .env에서 설정)
_default_unity = _BASE.parent.parent.parent / "unity-client" / "AURA_Lego" / "Assets" / "External Assets" / "capstone1" / "3d_model"
UNITY_ASSETS_PATH: Path = Path(os.getenv("UNITY_ASSETS_PATH", str(_default_unity)))

STORAGE_CAPTURES.mkdir(parents=True, exist_ok=True)
STORAGE_MODELS.mkdir(parents=True, exist_ok=True)
UNITY_ASSETS_PATH.mkdir(parents=True, exist_ok=True)
