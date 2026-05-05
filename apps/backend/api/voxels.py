"""
복셀 데이터 REST 엔드포인트.

engine-vision(charuco 파이프라인)이 JSON을 POST하면
상태에 저장하고, 연결된 Unity 클라이언트에 웹소켓으로 즉시 전송한다.

엔드포인트
----------
POST /voxels/upload  — engine-vision → backend
GET  /voxels/latest  — (폴링 폴백) Unity 또는 디버깅용
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from ..core import state

router = APIRouter(prefix="/voxels", tags=["voxels"])


class VoxelPayload(BaseModel):
    version: str
    timestamp: str
    board: dict[str, Any]
    voxels: list[dict[str, Any]]
    meta: dict[str, Any]


@router.post("/upload")
async def upload_voxels(payload: VoxelPayload):
    """
    engine-vision 파이프라인에서 복셀 데이터를 수신한다.
    저장 후 Unity 웹소켓으로 즉시 브로드캐스트한다.
    """
    async with state.lock:
        state.latest_voxels = payload.model_dump()

    msg = {
        "type":    "voxel_update",
        "payload": state.latest_voxels,
    }
    sent = await state.broadcast_to_unity(msg)
    return {
        "status":        "ok",
        "voxels":        len(payload.voxels),
        "unity_clients": sent,
    }


@router.get("/latest")
async def get_latest_voxels():
    """최신 복셀 데이터를 반환한다 (Unity 폴링 폴백)."""
    if state.latest_voxels is None:
        return {"error": "아직 데이터 없음"}, 404
    return state.latest_voxels
