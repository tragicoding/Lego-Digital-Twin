import os
from pathlib import Path
from dotenv import load_dotenv

# 어느 디렉토리에서 실행하든 .env를 찾을 수 있도록 절대경로 사용
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE)

DATABASE_URL: str = os.environ["DATABASE_URL"]
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
TRIPO_API_KEY: str = os.environ["TRIPO_API_KEY"]
STORAGE_BASE: Path = Path(os.getenv("STORAGE_BASE", "/app/storage"))
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "http://localhost:8000")
TRIPO_MODEL_VERSION: str = os.getenv("TRIPO_MODEL_VERSION", "P1-20260311")
TRIPO_DEFAULT_FACE_LIMIT: int = int(os.getenv("TRIPO_DEFAULT_FACE_LIMIT", "4000"))
TRIPO_DEFAULT_TEXTURE_QUALITY: str = os.getenv(
    "TRIPO_DEFAULT_TEXTURE_QUALITY",
    "standard",
)

STORAGE_IMAGES: Path = STORAGE_BASE / "images"
STORAGE_MODELS: Path = STORAGE_BASE / "models"

STORAGE_IMAGES.mkdir(parents=True, exist_ok=True)
STORAGE_MODELS.mkdir(parents=True, exist_ok=True)

UNITY_GENERATED_MODELS: Path = Path(
    os.getenv(
        "UNITY_GENERATED_MODELS",
        "/mnt/d/Lego-Digital-Twin/unity-client/AURA_Lego/Assets/StreamingAssets/GeneratedModels"
    )
)
UNITY_GENERATED_MODELS.mkdir(parents=True, exist_ok=True)

TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi"
RQ_QUEUE_NAME = "lego-queue"  # 레거시 — 하위 호환용
RQ_QUEUE_CHARACTER = "lego-character"
RQ_QUEUE_OBJECT = "lego-object"
