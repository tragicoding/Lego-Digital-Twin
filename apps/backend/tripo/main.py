"""
백엔드 서버의 메인 진입 파일
하드웨어, Tripo API, Unity, WebSocket을 연결하는 중앙 서버 역할

FastAPI app 생성
↓
CORS 허용 설정
↓
hardware / tripo / unity / websocket 라우터 등록
↓
/health API 제공
↓
React, Arduino, Windows 카메라, Unity가 이 서버를 통해 연결됨

"""



"""
Lego Digital Twin — 중앙 백엔드 서버

실행:
    conda run -n triposr uvicorn apps.backend.tripo.main:app --host 0.0.0.0 --port 8000 --reload
    또는 (tripo 디렉토리에서)
    conda run -n triposr uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    triposr conda 환경에서
    uvicorn으로
    main.py 안의 app 객체를 실행하고
    8000번 포트로 서버를 연다

데이터 흐름:
    Windows 카메라  → POST /hardware/camera/capture
                    → (자동) Tripo v3.0 API → GLB 생성
                    → Unity WebSocket 알림 + External Assets 복사
    Arduino         → POST /hardware/event    → Unity WebSocket
    Unity           ↔ WS  /ws/unity
"""

from fastapi import FastAPI
#프론트엔드 React가 localhost:5173, 백엔드가 localhost:8000처럼
#서로 다른 주소에서 실행되면 브라우저가 요청을 막을 수 있어.
#이걸 허용하기 위해 사용
from fastapi.middleware.cors import CORSMiddleware

#각 기능별 API 라우터를 가져온다.
from .api.routers.hardware import router as hardware_router
from .api.routers.tripo import router as tripo_router
from .api.routers.unity import router as unity_router
from .ws.unity_ws import router as ws_router


#FastAPI 서버 객체를 생성
#여기서 설정한 title, description, version은 APi문서에 표시됨.
#서버 실행후 아래 주소에서 확인 가능
#http://localhost:8000/docs
app = FastAPI(
    title="Lego Digital Twin Backend",
    description="Hardware / Tripo3D / Unity 데이터 중계 서버",
    version="2.0.0",
)

"""0
프론트엔드나 Unity 등 외부 
클라이언트가 백엔드에 요청할 수 있도록 허용하는 설정
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #모든 출처에서 요청 허용
    allow_methods=["*"], #모든 HTTP 메서드 허용 (GET, POST, etc.)
    allow_headers=["*"], #모든 헤더 허용.
    #개발 단계에서는 편하지만, 실제 배포에서는 특정 주소만 허용.
)

#라우터 등록.
#각 기능별 API를 메인 FastAPI 앱에 연결
app.include_router(hardware_router)
app.include_router(tripo_router)
app.include_router(unity_router)
app.include_router(ws_router)


@app.get("/health", tags=["health"])
async def health():
    from .ws.unity_ws import manager
    return {"status": "ok", "unity_clients": manager.count}
