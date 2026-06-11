"""
Python + FastAPI + SQLAlchemy로 작성된 에셋 업로드/상태 조회 라우터
프론트엔드에서 업로드한 이미지 파일을 저장하고, DB에 등록한 뒤,
 Redis Queue에 작업을 넣는 역할

 [CapturePage]
사진 업로드
↓
POST /sessions/{session_id}/assets
↓
서버가 이미지 파일 저장
↓
DB에 Asset 생성
↓
Redis Queue에 asset.id 등록
↓
Worker가 3D 변환 처리
↓
DB에 status/progress/model_url 업데이트
↓
[LoadingPage]
GET /sessions/{session_id}/status를 3초마다 호출
↓
진행률 표시
↓
모든 asset 완료 + 프로필 완료
↓
ready_for_unity = true

"""


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
from ...services.queue_service import enqueue_asset, enqueue_object_session
from ...services import session_queue_service as sq

#기본 경로는 /sessions
"""
POST /sessions/{session_id}/assets
GET  /sessions/{session_id}/status 과 같은 API가 만들어진다.
즉, 프론트엔드의 다음 코드와 연결
uploadAsset(sessionId, step.type, selectedFile)
// POST /sessions/{sessionId}/assets
getStatus(sessionId)
// GET /sessions/{sessionId}/status
"""
router = APIRouter(prefix="/sessions", tags=["assets"])

ASSET_TYPES = {"character", "building", "vehicle"}

#이미지 업로드 API
#프론트엔드에서 FormData로 보낸 데이터가 여기로 들어온다.
"""
form.append("asset_type", assetType);
form.append("file", file);
"""
@router.post("/{session_id}/assets", response_model=AssetUploadResponse, status_code=201)
async def upload_asset(
    session_id: str,
    asset_type: Literal["character", "building", "vehicle", "object"] = Form(...),
    file: UploadFile = File(...),
    # view: front(기본, 파이프라인 시작) | left/back/right(파일 저장만, 큐 미등록)
    view: Literal["front", "back", "left", "right"] = Form("front"),
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    queued_session = sq.get_session_data(session_id)
    if not session and not queued_session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    if session and session.status == "cancelled":
        raise HTTPException(409, "취소된 세션에는 이미지를 업로드할 수 없습니다.")

    save_dir = STORAGE_IMAGES / session_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # 심사용 플로우: 캐릭터는 사전 제작 FBX를 관리자에서 등록하므로 촬영만 통과시킨다.
    if asset_type == "character":
        save_path = save_dir / f"character_{view}.jpg"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return AssetUploadResponse(asset_id="", status="saved", asset_type=f"character_{view}")

    # 심사용 플로우: Redis session_queue 세션은 오브제 파일만 저장하고 object worker에 session_id를 보낸다.
    if queued_session and asset_type == "object":
        save_path = save_dir / f"object_{view}.jpg"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if view == "front":
            sq.update_session(session_id, object_status="queued", object_stage="waiting", object_progress=0)
            enqueue_object_session(session_id)
        return AssetUploadResponse(asset_id="", status="queued" if view == "front" else "saved", asset_type=f"object_{view}")

    # left/back/right 이미지: 파일 저장만, DB/큐 등록 없음 (worker가 polling으로 감지)
    if view in ("back", "left", "right") and asset_type in ("character", "object"):
        save_path = save_dir / f"{asset_type}_{view}_{file.filename}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return AssetUploadResponse(
            asset_id="",
            status="saved",
            asset_type=f"{asset_type}_{view}",
        )

    # front / 오브제: 기존 파이프라인 그대로
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
    #작업 큐에 등록해서 백그라운드 워커가 처리하게 하는 구조
    enqueue_asset(asset.id, asset_type)
    """
    이미지 업로드
↓
DB에 Asset 저장
↓
Redis Queue에 asset.id 등록
↓
Worker가 Queue에서 작업 가져감
↓
Tripo API 호출
↓
GLB 생성
↓
DB 상태 업데이트"""


    return AssetUploadResponse(
        asset_id=asset.id,
        status="queued",
        asset_type=asset_type,
    )
    """
    프론트엔드에 업로드 결과 반환
    {
  "asset_id": 1,
  "status": "queued",
  "asset_type": "character"
}"""


#상태 조회 API
#특정 세션의 처리 상태를 조회하는 API
#프론트엔드 LoadingPage에서 3초마다 호출하는 함수와 연결
#const res = await getStatus(sessionID);
@router.get("/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str, db: DBSession = Depends(get_db)):
    #session_id 한번더 확인
    session = db.get(Session, session_id)
    queued_session = sq.get_session_data(session_id)
    if queued_session:
        object_status = queued_session.get("object_status", "waiting")
        object_stage = queued_session.get("object_stage", "waiting")
        object_progress = int(queued_session.get("object_progress") or 0)
        character_ready = bool(queued_session.get("character_no"))
        ready_for_unity = character_ready and object_status == "completed"
        return SessionStatusResponse(
            session_id=session_id,
            profile_completed=character_ready,
            assets={
                "character": AssetStatusItem(
                    status="completed" if character_ready else "waiting",
                    stage="ready" if character_ready else "waiting",
                    progress=100 if character_ready else 0,
                    model_url=queued_session.get("character_model_url") or None,
                ),
                "object": AssetStatusItem(
                    status=object_status,
                    stage=object_stage,
                    progress=object_progress,
                    model_url=queued_session.get("object_model_url") or None,
                ),
            },
            ready_for_unity=ready_for_unity,
        )
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    #각 asset 상태 모으기
    #프론트엔드로 보낼 에셋 상태 객체.
    assets_status: dict[str, AssetStatusItem] = {}
    all_completed = True
    """
    {
  "character": {
    "status": "completed",
    "stage": "ready",
    "progress": 100,
    "model_url": "/models/character.glb"
  },
  "building": {
    "status": "processing",
    "stage": "generating",
    "progress": 60,
    "model_url": null
  }
}"""
    #해당 세션에 연결된 모든 asset을 반복.
    #각 asset의 상태를 프론트엔드가 읽기 좋은 형태로 정리
    #Session 모델과 Asset 모델이 연결되어 있는 구조
    for asset in session.assets:
        assets_status[asset.asset_type] = AssetStatusItem(
            status=asset.status,
            stage=asset.stage,
            progress=asset.progress,
            model_url=asset.model_url,
        )
        #하나라도 완료되지 않은 asset이 있으면 전체 완료가 아니라고 판단
        if asset.status != "completed":
            all_completed = False
    #Unity로 넘길 준비가 되어있는지 판단하는 조건. 
    #조건은 3개
    #1. asset이 하나 이상 있어야 함
    #2. 모든 asset 처리가 completed여야 함
    #3. nickname이 있어야 함, 즉 프로필 입력 완료
    #즉, 이미지 변환만 끝났다고 바로 Unity 준비 완료가 아니고,
    #프로필까지 입력되어야 ready_for_unity = True가 된다.
    has_assets = len(session.assets) > 0
    ready_for_unity = (
        session.status != "cancelled"
        and has_assets
        and all_completed
        and bool(session.nickname)
    )

    return SessionStatusResponse(
        session_id=session_id,
        profile_completed=bool(session.nickname),
        assets=assets_status,
        ready_for_unity=ready_for_unity,
    )
"""
예시응답:
{
  "session_id": "abc123",
  "profile_completed": true,
  "assets": {
    "character": {
      "status": "completed",
      "stage": "ready",
      "progress": 100,
      "model_url": "/models/character.glb"
    }
  },
  "ready_for_unity": true
}"""
