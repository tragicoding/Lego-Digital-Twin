"""
Lego Digital Twin 중앙 백엔드 서버.

실행 방법
---------
  conda run -n charuco uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000 --reload

통신 구조
---------
  engine-vision (charuco) → POST /voxels/upload  → 상태 저장 + Unity WS 브로드캐스트
  hardware (Arduino)      → POST /hardware/event → Unity WS 브로드캐스트
  Unity                   ↔  WS  /ws/unity       ← 이벤트 수신

API 문서
--------
  http://localhost:8000/docs  (Swagger UI 자동 생성)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.voxels   import router as voxels_router
from .api.hardware import router as hardware_router
from .ws.unity_ws  import router as ws_router

app = FastAPI(
    title="Lego Digital Twin Backend",
    description="vision / hardware / Unity 사이의 데이터 중계 서버",
    version="0.1.0",
)

# 개발 환경에서 CORS 전체 허용 (Unity WebGL 빌드 대비)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voxels_router)
app.include_router(hardware_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    """서버 상태 확인."""
    from .core.state import unity_connections
    return {"status": "ok", "unity_clients": len(unity_connections)}
