"""
Hardware 라우터

엔드포인트:
    POST /hardware/camera/capture  — Windows 카메라에서 이미지 업로드
                                     → Tripo v3.0으로 3D 변환 비동기 시작
                                     → task_id 반환
    POST /hardware/event           — Arduino 이벤트 수신 → Unity WebSocket 전달
    GET  /hardware/status          — 최신 조명 상태 조회
"""

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Literal

from ...core.config import TRIPO_API_KEY, STORAGE_CAPTURES, STORAGE_MODELS
from ...services.tripo_service import TripoService
from ...services.unity_service import deliver_to_unity
from ...ws.unity_ws import manager

router = APIRouter(prefix="/hardware", tags=["hardware"])

_tripo = TripoService(TRIPO_API_KEY)

# 인메모리 task 상태 저장소 (tripo 라우터와 공유)
from ...api.routers import _task_store as tasks

# 조도 → atmosphere 매핑
_LUX_MAP = [(2000, "day"), (800, "cloudy"), (200, "sunset"), (50, "night"), (0, "midnight")]


def _lux_to_atmosphere(lux: float) -> str:
    for threshold, label in _LUX_MAP:
        if lux >= threshold:
            return label
    return "midnight"


_latest_light: dict = {"lux": 1000, "atmosphere": "day"}


class HardwareEvent(BaseModel):
    type: Literal["clap", "light"]
    value: float = 0.0


@router.post("/camera/capture")
async def camera_capture(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Windows 카메라 클라이언트가 이미지를 POST → Tripo 3D 변환 파이프라인 시작.
    즉시 task_id를 반환하고, 변환은 백그라운드에서 진행된다.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 가능합니다.")

    task_id = str(uuid.uuid4())
    image_bytes = await file.read()

    # 캡처 이미지 저장
    capture_path = STORAGE_CAPTURES / file.filename
    capture_path.write_bytes(image_bytes)

    tasks[task_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(
        _run_tripo_pipeline, task_id, image_bytes, file.filename
    )
    return {"task_id": task_id, "status": "queued", "filename": file.filename}


async def _run_tripo_pipeline(task_id: str, image_bytes: bytes, filename: str):
    tasks[task_id]["status"] = "processing"
    try:
        stem = Path(filename).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = STORAGE_MODELS / f"{stem}_{ts}.glb"

        glb_path = await _tripo.run(image_bytes, filename, output_path)

        # Unity External Assets에 복사
        unity_path = deliver_to_unity(glb_path)

        tasks[task_id] = {
            "status": "success",
            "filename": glb_path.name,
            "download_url": f"/unity/models/{glb_path.name}",
            "unity_path": str(unity_path),
        }

        # Unity에 WebSocket으로 알림
        await manager.broadcast({
            "type": "model_ready",
            "filename": glb_path.name,
            "download_url": f"/unity/models/{glb_path.name}",
        })

    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}
        await manager.broadcast({"type": "model_failed", "task_id": task_id, "error": str(e)})


@router.post("/event")
async def hardware_event(event: HardwareEvent):
    """Arduino 이벤트 수신 → Unity WebSocket 브로드캐스트."""
    global _latest_light

    if event.type == "clap":
        msg = {"type": "fireworks", "intensity": event.value}

    elif event.type == "light":
        atm = _lux_to_atmosphere(event.value)
        _latest_light = {"lux": event.value, "atmosphere": atm}
        msg = {"type": "atmosphere", "atmosphere": atm, "lux": event.value}

    else:
        return {"status": "unknown event"}

    sent = await manager.broadcast(msg)
    return {"status": "ok", "unity_clients": sent, "message": msg}


@router.get("/status")
async def hardware_status():
    """현재 조명 상태 반환."""
    return _latest_light
