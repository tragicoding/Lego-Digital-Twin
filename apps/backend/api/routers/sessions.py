"""
세션 라우터
POST /sessions                        — 세션 생성 (관객 입장 즉시)
PATCH /sessions/{id}/profile          — 닉네임/말풍선 입력
PATCH /sessions/{id}/names            — Unity 대기화면에서 입력한 이름 저장
GET  /sessions/{id}                   — 세션 정보 조회
POST /sessions/{id}/like              — 좋아요 +1 (실시간 WebSocket 브로드캐스트)
GET  /sessions/active                 — Unity 대기 큐 맨 앞 세션 조회
GET  /sessions/unity-queue            — Unity 대기 큐 전체 목록
POST /sessions/unity-queue/advance    — Unity 현재 세션 완료 → 다음 세션으로 전진
DELETE /sessions/unity-queue/{id}     — 대기 큐에서 특정 세션 제거 (관리자)
"""
from fastapi import APIRouter, Depends, HTTPException
import httpx
import uuid
from sqlalchemy.orm import Session as DBSession

from ...core.config import BACKEND_HOST, WINDOWS_CAMERA_URL
from ...core.database import get_db
from ...models.asset import Asset
from ...models.session import Session
from ...schemas.session import (
    SessionCreateResponse,
    ProfileUpdate,
    SessionResponse,
    SignatureMotionUpdate,
)
from ...schemas.unity import LikeResponse
from ...services import unity_queue_service as uq

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _next_numeric_session_id(db: DBSession) -> str:
    numeric_ids = []
    for (session_id,) in db.query(Session.id).all():
        if session_id.isdigit():
            numeric_ids.append(int(session_id))
    return str(max(numeric_ids) + 1 if numeric_ids else 0)


def _character_number(bottom: int, middle: int, top: int) -> int:
    return (bottom - 1) * 9 + (middle - 1) * 3 + top


def _get_or_create_character_asset(db: DBSession, session_id: str) -> Asset:
    asset = (
        db.query(Asset)
        .filter(Asset.session_id == session_id, Asset.asset_type == "character")
        .first()
    )
    if asset:
        return asset

    asset = Asset(
        session_id=session_id,
        asset_type="character",
        status="waiting_admin",
        stage="admin_selection",
        progress=50,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


@router.post("", response_model=SessionCreateResponse, status_code=201)
def create_session(db: DBSession = Depends(get_db)):
    """앱 시작 시 DB에 실제 관람객 세션을 생성한다."""
    session_id = _next_numeric_session_id(db)
    while db.get(Session, session_id):
        session_id = str(int(session_id) + 1)

    db.add(Session(id=session_id, status="active"))
    db.commit()
    return SessionCreateResponse(session_id=session_id)


@router.post("/{session_id}/character-placeholder")
def create_character_placeholder(session_id: str, db: DBSession = Depends(get_db)):
    """캐릭터는 실제 촬영/Tripo 없이 관리자 조합 선택 대기 상태로 등록한다."""
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    if session.status == "cancelled":
        raise HTTPException(409, "취소된 세션입니다.")

    asset = _get_or_create_character_asset(db, session_id)
    if session.character_number:
        asset.status = "completed"
        asset.stage = "ready"
        asset.progress = 100
    else:
        asset.status = "waiting_admin"
        asset.stage = "admin_selection"
        asset.progress = 50
    db.commit()
    return {"status": asset.status, "session_id": session_id, "asset_id": asset.id}


@router.post("/{session_id}/capture/object")
def capture_object_from_windows_camera(session_id: str, db: DBSession = Depends(get_db)):
    """Windows USB camera service에 4방향 오브제 촬영과 업로드를 요청한다."""
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    if session.status == "cancelled":
        raise HTTPException(409, "취소된 세션입니다.")

    url = f"{WINDOWS_CAMERA_URL.rstrip('/')}/capture-and-upload"
    capture_id = f"{session_id}_{uuid.uuid4().hex[:12]}"
    payload = {
        "backend_url": BACKEND_HOST,
        "session_id": session_id,
        "asset_type": "object",
        "capture_id": capture_id,
    }
    try:
        response = httpx.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Windows camera service 호출 실패: {exc}",
        ) from exc

    try:
        return response.json()
    except ValueError:
        return {"status": "ok", "camera_response": response.text}


@router.patch("/{session_id}/profile")
def update_profile(
    session_id: str,
    body: ProfileUpdate,
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    if body.character_npc_name:
        session.nickname = body.character_npc_name
    elif body.nickname:
        session.nickname = body.nickname

    if body.object_name:
        session.object_name = body.object_name

    if body.bubble_text:
        session.bubble_text = body.bubble_text

    if body.phone:
        session.phone = body.phone
    if body.favorite_theme:
        session.favorite_theme = body.favorite_theme

    db.commit()

    return {"status": "ok", "session_id": session_id}


@router.patch("/{session_id}/names")
def update_names(
    session_id: str,
    body: ProfileUpdate,
    db: DBSession = Depends(get_db),
):
    """Unity 대기화면에서 받은 캐릭터/오브제 이름을 저장한다."""
    return update_profile(session_id, body, db)


@router.get("/active")
def get_active_session(db: DBSession = Depends(get_db)):
    """Unity 게임 시작 시 unity_queue 맨 앞을 runtime으로 pop한다."""
    active_id = uq.get_active_session_id()
    if active_id:
        active_session = db.get(Session, active_id)
        if active_session and active_session.status not in ("cancelled", "completed"):
            return {"session_id": active_id}
        uq.remove_from_all_queues(active_id)

    session_id = uq.pop_unity_for_runtime()
    if session_id:
        session = db.get(Session, session_id)
        if session and session.status != "cancelled":
            session.status = "runtime"
            db.commit()
    return {"session_id": session_id}


@router.get("/unity-queue")
def get_unity_queue():
    """Unity 대기 큐 전체 목록을 순서대로 반환한다 (관리자용)."""
    return {"queue": uq.list_queue(uq.UNITY_QUEUE_KEY)}


@router.post("/unity-queue/advance")
def advance_unity_queue(db: DBSession = Depends(get_db)):
    """Unity 체험 완료 시 active 세션을 history_queue로 보낸다."""
    active = uq.get_active_session_id()
    if active:
        uq.move_active_to_history(active)
        session = db.get(Session, active)
        if session and session.status != "cancelled":
            session.status = "completed"
            db.commit()
    return {"removed": active, "next_session_id": None}


@router.delete("/unity-queue/{session_id}")
def remove_from_unity_queue(session_id: str):
    """대기 큐에서 특정 세션을 제거한다 (관리자용)."""
    uq.remove_from_all_queues(session_id)
    return {"status": "ok", "removed": session_id}


@router.post("/{session_id}/cancel")
def cancel_session(session_id: str, db: DBSession = Depends(get_db)):
    """운영자용 취소 처리.

    취소된 세션은 추가 업로드를 막고 unity_queue에서도 제거한다.
    실행 중인 worker는 다음 단계 체크에서 중단된다.
    """
    session = db.get(Session, session_id)
    uq.remove_from_all_queues(session_id)
    if not session:
        return {"status": "ok", "session_id": session_id, "removed_from_queues": True}

    session.status = "cancelled"
    for asset in session.assets:
        if asset.status not in ("completed", "failed", "cancelled"):
            asset.status = "cancelled"
            asset.stage = "cancelled"
    db.commit()

    return {
        "status": "ok",
        "session_id": session_id,
        "removed_from_queues": True,
    }


@router.patch("/{session_id}/signature-motion")
def update_signature_motion(
    session_id: str,
    body: SignatureMotionUpdate,
    db: DBSession = Depends(get_db),
):
    """Unity 가이드 모드 종료 시 관람객이 선택한 시그니처 동작 저장."""
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    session.signature_motion = body.signature_motion
    db.commit()
    return {"status": "ok", "session_id": session_id, "signature_motion": body.signature_motion}


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    return session


@router.post("/{session_id}/like", response_model=LikeResponse)
def like_session(session_id: str, db: DBSession = Depends(get_db)):
    """좋아요 +1. 완료된 세션만 좋아요 가능. 실시간 WebSocket 브로드캐스트."""
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    session.likes = (session.likes or 0) + 1
    db.commit()

    # 전체 중 좋아요 1위 세션 계산
    from sqlalchemy import desc
    top = db.query(Session).order_by(desc(Session.likes)).first()
    top_session_id = top.id if top else None

    # WebSocket 브로드캐스트 (Redis pub/sub)
    from ...services.event_service import publish_likes_updated
    publish_likes_updated(session_id, session.likes, top_session_id)

    return LikeResponse(
        session_id=session_id,
        likes=session.likes,
        is_top_liked=(top_session_id == session_id),
        top_session_id=top_session_id,
    )
