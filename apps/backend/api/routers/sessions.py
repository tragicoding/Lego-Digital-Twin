"""
세션 라우터
POST /sessions                        — 세션 생성 (관객 입장 즉시)
PATCH /sessions/{id}/profile          — 닉네임/말풍선 입력
GET  /sessions/{id}                   — 세션 정보 조회
POST /sessions/{id}/like              — 좋아요 +1 (실시간 WebSocket 브로드캐스트)
GET  /sessions/active                 — Unity 대기 큐 맨 앞 세션 조회
GET  /sessions/unity-queue            — Unity 대기 큐 전체 목록
POST /sessions/unity-queue/advance    — Unity 현재 세션 완료 → 다음 세션으로 전진
DELETE /sessions/unity-queue/{id}     — 대기 큐에서 특정 세션 제거 (관리자)
"""
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from redis import Redis

from ...core.config import REDIS_URL
from ...core.database import get_db
from ...models.session import Session
from ...schemas.session import SessionCreateResponse, ProfileUpdate, SessionResponse, SignatureMotionUpdate
from ...schemas.unity import LikeResponse
from ...services.event_service import UNITY_QUEUE_KEY

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _next_session_id(db: DBSession) -> str:
    """숫자 4자리 형식(0000~9999) ID 중 최댓값 +1을 반환한다."""
    rows = db.query(Session.id).all()
    numeric_ids = [int(sid) for (sid,) in rows if re.fullmatch(r"\d{4}", sid)]
    next_num = (max(numeric_ids) + 1) if numeric_ids else 0
    return f"{next_num:04d}"


@router.post("", response_model=SessionCreateResponse, status_code=201)
def create_session(db: DBSession = Depends(get_db)):
    session = Session(id=_next_session_id(db))
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionCreateResponse(session_id=session.id)


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
        session.bubble_text = body.object_name
    elif body.bubble_text:
        session.bubble_text = body.bubble_text

    if body.phone:
        session.phone = body.phone
    if body.favorite_theme:
        session.favorite_theme = body.favorite_theme

    db.commit()

    # Worker가 profile 설정 전에 완료된 경우를 대비해 재확인
    from ...services.event_service import check_and_notify
    check_and_notify(session_id)

    return {"status": "ok", "session_id": session_id}


@router.get("/active")
def get_active_session():
    """Unity가 지금 표시해야 하는 세션 ID를 반환한다. 큐가 비어있으면 null."""
    r = Redis.from_url(REDIS_URL)
    val = r.lindex(UNITY_QUEUE_KEY, 0)
    r.close()
    return {"session_id": val.decode() if val else None}


@router.get("/unity-queue")
def get_unity_queue():
    """Unity 대기 큐 전체 목록을 순서대로 반환한다 (관리자용)."""
    r = Redis.from_url(REDIS_URL)
    items = r.lrange(UNITY_QUEUE_KEY, 0, -1)
    r.close()
    return {"queue": [item.decode() for item in items]}


@router.post("/unity-queue/advance")
def advance_unity_queue():
    """Unity에서 현재 세션 체험 완료 시 호출 — 큐 맨 앞을 제거하고 다음 세션을 반환한다."""
    r = Redis.from_url(REDIS_URL)
    popped = r.lpop(UNITY_QUEUE_KEY)
    next_val = r.lindex(UNITY_QUEUE_KEY, 0)
    r.close()
    return {
        "removed": popped.decode() if popped else None,
        "next_session_id": next_val.decode() if next_val else None,
    }


@router.delete("/unity-queue/{session_id}")
def remove_from_unity_queue(session_id: str):
    """대기 큐에서 특정 세션을 제거한다 (관리자용)."""
    r = Redis.from_url(REDIS_URL)
    removed_count = r.lrem(UNITY_QUEUE_KEY, 0, session_id)
    r.close()
    if removed_count == 0:
        raise HTTPException(404, "큐에서 해당 세션을 찾을 수 없습니다.")
    return {"status": "ok", "removed": session_id}


@router.post("/{session_id}/finalize")
def finalize_session(session_id: str, db: DBSession = Depends(get_db)):
    """프론트 키오스크에서 새 관람객 시작 전 호출.

    이미 완료된 세션이면 unity_queue 등록을 재확인하고, 아직 미완료면 현 상태만 반환한다.
    서버가 큐의 정본이 되도록 중복 등록은 Redis에서 방지한다.
    """
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    from ...services.event_service import check_and_notify

    check_and_notify(session_id)

    ready = session.status != "cancelled" and bool(session.nickname) and bool(session.assets) and all(
        asset.status == "completed" for asset in session.assets
    )

    r = Redis.from_url(REDIS_URL)
    try:
        queue = [item.decode() for item in r.lrange(UNITY_QUEUE_KEY, 0, -1)]
    finally:
        r.close()

    return {
        "status": "ok",
        "session_id": session_id,
        "ready_for_unity": ready,
        "queued": session_id in queue,
        "queue_position": (queue.index(session_id) + 1) if session_id in queue else None,
    }


@router.post("/{session_id}/cancel")
def cancel_session(session_id: str, db: DBSession = Depends(get_db)):
    """운영자용 취소 처리.

    취소된 세션은 추가 업로드를 막고 unity_queue에서도 제거한다.
    실행 중인 worker는 다음 단계 체크에서 중단된다.
    """
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    session.status = "cancelled"
    for asset in session.assets:
        if asset.status not in ("completed", "failed", "cancelled"):
            asset.status = "cancelled"
            asset.stage = "cancelled"
    db.commit()

    r = Redis.from_url(REDIS_URL)
    try:
        removed_count = r.lrem(UNITY_QUEUE_KEY, 0, session_id)
    finally:
        r.close()

    return {
        "status": "ok",
        "session_id": session_id,
        "removed_from_unity_queue": removed_count,
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
