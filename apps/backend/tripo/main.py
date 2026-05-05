"""
Lego Digital Twin — 중앙 백엔드 서버

실행:
    conda run -n triposr uvicorn apps.backend.tripo.main:app --host 0.0.0.0 --port 8000 --reload
    또는 (tripo 디렉토리에서)
    conda run -n triposr uvicorn main:app --host 0.0.0.0 --port 8000 --reload

데이터 흐름:
    Windows 카메라  → POST /hardware/camera/capture
                    → (자동) Tripo v3.0 API → GLB 생성
                    → Unity WebSocket 알림 + External Assets 복사
    Arduino         → POST /hardware/event    → Unity WebSocket
    Unity           ↔ WS  /ws/unity
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers.hardware import router as hardware_router
from .api.routers.tripo import router as tripo_router
from .api.routers.unity import router as unity_router
from .ws.unity_ws import router as ws_router

app = FastAPI(
    title="Lego Digital Twin Backend",
    description="Hardware / Tripo3D / Unity 데이터 중계 서버",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hardware_router)
app.include_router(tripo_router)
app.include_router(unity_router)
app.include_router(ws_router)


@app.get("/health", tags=["health"])
async def health():
    from .ws.unity_ws import manager
    return {"status": "ok", "unity_clients": manager.count}
