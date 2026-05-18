"""
테스트 전용 라우터 — test_app 브랜치에서만 사용.
실제 이미지 없이 가짜 데이터로 전체 파이프라인을 테스트한다.

POST /test/start
  → 세션 생성 + 캐릭터/건물/자동차 3개 Mock asset 등록 + Worker 시작
  → { session_id, asset_ids }

POST /test/sessions/{session_id}/profile
  → 닉네임/말풍선 자동 입력
"""
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session as DBSession

from ...core.database import get_db
from ...models.asset import Asset
from ...models.session import Session
from ...services.queue_service import enqueue_mock_asset

router = APIRouter(prefix="/test", tags=["test"])


@router.post("/start")
def test_start(
    background_tasks: BackgroundTasks,
    nickname: str = "테스트유저",
    bubble_text: str = "안녕하세요! 제 레고 월드에 오신 걸 환영해요 🧱",
    db: DBSession = Depends(get_db),
):
    """
    세션 생성 + 프로필 입력 + 3개 Mock asset 등록을 한 번에 처리.
    Worker가 Mock 파이프라인을 병렬로 실행한다.
    """
    # 세션 생성 + 프로필 바로 입력
    session = Session(nickname=nickname, bubble_text=bubble_text)
    db.add(session)
    db.commit()
    db.refresh(session)

    # Mock asset 3개 생성 — 각 타입 전용 큐에 등록되어 병렬 처리됨
    asset_ids = {}
    for asset_type in ("character", "building", "vehicle"):
        asset = Asset(
            session_id=session.id,
            asset_type=asset_type,
            input_image_path=f"mock/{asset_type}_placeholder.png",
            status="queued",
            stage="waiting",
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        asset_ids[asset_type] = asset.id
        enqueue_mock_asset(asset.id, asset_type)

    return {
        "session_id": session.id,
        "nickname": nickname,
        "asset_ids": asset_ids,
        "status_url": f"/sessions/{session.id}/status",
        "unity_url": f"/unity/sessions/{session.id}",
        "message": "Mock 파이프라인 시작. /sessions/{session_id}/status 로 상태 확인",
    }


@router.post("/sessions/{session_id}/retry/{asset_type}")
def retry_asset(session_id: str, asset_type: str, db: DBSession = Depends(get_db)):
    """실패한 에셋을 재시도한다 (실제 파이프라인 재실행)."""
    from ...models.asset import Asset
    from ...services.queue_service import enqueue_asset

    asset = db.query(Asset).filter(
        Asset.session_id == session_id,
        Asset.asset_type == asset_type,
    ).first()
    if not asset:
        return {"status": "not found"}

    asset.status = "queued"
    asset.stage = "waiting"
    asset.progress = 0
    asset.error_message = None
    db.commit()

    enqueue_asset(asset.id, asset_type)
    return {"status": "requeued", "asset_id": asset.id}


@router.delete("/sessions/{session_id}")
def test_cleanup(session_id: str, db: DBSession = Depends(get_db)):
    """테스트 세션 삭제 (초기화용)."""
    session = db.get(Session, session_id)
    if not session:
        return {"status": "not found"}
    for asset in session.assets:
        for anim in asset.animations:
            db.delete(anim)
        db.delete(asset)
    db.delete(session)
    db.commit()
    return {"status": "deleted", "session_id": session_id}
