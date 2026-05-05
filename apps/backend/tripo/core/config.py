import os
from pathlib import Path
from dotenv import load_dotenv

_BASE = Path(__file__).parent.parent
load_dotenv(_BASE / ".env")

TRIPO_API_KEY: str = os.environ["TRIPO_API_KEY"]
TRIPO_BASE_URL: str = "https://api.tripo3d.ai/v2/openapi"

STORAGE_CAPTURES: Path = _BASE / "storage" / "captures"
STORAGE_MODELS: Path = _BASE / "storage" / "models"

STORAGE_CAPTURES.mkdir(parents=True, exist_ok=True)
STORAGE_MODELS.mkdir(parents=True, exist_ok=True)
