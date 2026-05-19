import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ["DATABASE_URL"]
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
TRIPO_API_KEY: str = os.environ["TRIPO_API_KEY"]
STORAGE_BASE: Path = Path(os.getenv("STORAGE_BASE", "/app/storage"))
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "http://localhost:8000")

STORAGE_IMAGES: Path = STORAGE_BASE / "images"
STORAGE_MODELS: Path = STORAGE_BASE / "models"

STORAGE_IMAGES.mkdir(parents=True, exist_ok=True)
STORAGE_MODELS.mkdir(parents=True, exist_ok=True)

TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi"
RQ_QUEUE_NAME = "lego-queue"
