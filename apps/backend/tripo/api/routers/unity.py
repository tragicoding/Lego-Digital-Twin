"""
Unity 라우터

엔드포인트:
    GET /unity/models              — 생성된 GLB 모델 목록
    GET /unity/models/{filename}   — GLB 파일 다운로드
    GET /unity/health              — Unity 연결 수 확인
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...core.config import STORAGE_MODELS
from ...schemas.tripo import ModelListItem
from ...ws.unity_ws import manager

router = APIRouter(prefix="/unity", tags=["unity"])


@router.get("/models", response_model=list[ModelListItem])
async def list_models():
    """생성된 GLB 파일 목록을 반환한다."""
    models = []
    for f in sorted(STORAGE_MODELS.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
        models.append(ModelListItem(
            filename=f.name,
            download_url=f"/unity/models/{f.name}",
            unity_path=str(f),
            size_bytes=f.stat().st_size,
        ))
    return models


@router.get("/models/{filename}")
async def download_model(filename: str):
    """GLB 파일을 다운로드한다."""
    path = STORAGE_MODELS / filename
    if not path.exists():
        raise HTTPException(404, "파일이 없습니다.")
    return FileResponse(path, media_type="model/gltf-binary", filename=filename)


@router.get("/health")
async def unity_connection_status():
    return {"unity_clients": manager.count}
