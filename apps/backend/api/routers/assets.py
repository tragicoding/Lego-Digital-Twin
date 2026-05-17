"""
에셋 라우터
POST /sessions/{id}/assets   — 이미지 업로드 → Queue 등록
GET  /sessions/{id}/status   — 처리 상태 조회
"""
import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session as DBSession

from ...core.config import STORAGE_IMAGES
from ...core.database import get_db
from ...models.asset import Asset
from ...models.session import Session
from ...schemas.asset import AssetUploadResponse, SessionStatusResponse, AssetStatusItem
from ...services.queue_service import enqueue_asset

router = APIRouter(prefix="/sessions", tags=["assets"])

ASSET_TYPES = {"character", "building", "vehicle"}


@router.post("/{session_id}/assets", response_model=AssetUploadResponse, status_code=201)
async def upload_asset(
    session_id: str,
    asset_type: Literal["character", "building", "vehicle"] = Form(...),
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    # 이미지 저장
    save_dir = STORAGE_IMAGES / session_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{asset_type}_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # DB 등록
    asset = Asset(
        session_id=session_id,
        asset_type=asset_type,
        input_image_path=str(save_path),
        status="queued",
        stage="waiting",
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)

    # Redis Queue 등록
    enqueue_asset(asset.id, asset_type)

    return AssetUploadResponse(
        asset_id=asset.id,
        status="queued",
        asset_type=asset_type,
    )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    assets_status: dict[str, AssetStatusItem] = {}
    all_completed = True

    for asset in session.assets:
        assets_status[asset.asset_type] = AssetStatusItem(
            status=asset.status,
            stage=asset.stage,
            progress=asset.progress,
            model_url=asset.model_url,
        )
        if asset.status != "completed":
            all_completed = False

    has_assets = len(session.assets) > 0
    ready_for_unity = has_assets and all_completed and bool(session.nickname)

    return SessionStatusResponse(
        session_id=session_id,
        profile_completed=bool(session.nickname),
        assets=assets_status,
        ready_for_unity=ready_for_unity,
    )
