"""
Lego Digital Twin — 전시 백엔드 서버

실행:
    conda run -n triposr uvicorn main:app --host 0.0.0.0 --port 8000 --reload

API 문서:
    http://localhost:8000/docs
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import STORAGE_MODELS
from .api.routers.sessions import router as sessions_router
from .api.routers.assets import router as assets_router
from .api.routers.unity_data import router as unity_router
from .api.routers.ws_router import router as ws_router
from .services.ws_manager import manager
from .services.event_service import redis_subscriber


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Redis Pub/Sub → WebSocket 브로드캐스트 백그라운드 태스크
    task = asyncio.create_task(redis_subscriber(manager.broadcast))
    yield
    task.cancel()


app = FastAPI(
    title="Lego Digital Twin API",
    description="전시 관객 세션 관리 / Tripo 3D 파이프라인 / Unity 데이터 제공",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 생성된 GLB 파일 정적 서빙
app.mount("/static/models", StaticFiles(directory=str(STORAGE_MODELS)), name="models")

app.include_router(sessions_router)
app.include_router(assets_router)
app.include_router(unity_router)
app.include_router(ws_router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
