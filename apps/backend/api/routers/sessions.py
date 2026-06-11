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
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ...core.database import get_db
from ...models.session import Session
from ...schemas.session import SessionCreateResponse, ProfileUpdate, SessionResponse, SignatureMotionUpdate
from ...schemas.unity import LikeResponse
from ...services import session_queue_service as sq

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionCreateResponse, status_code=201)
def create_session():
    """앱 시작 시 DB 저장 없이 session_queue에 임시 세션을 추가한다."""
    return SessionCreateResponse(session_id=sq.create_session())


@router.patch("/{session_id}/profile")
def update_profile(
    session_id: str,
    body: ProfileUpdate,
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    if not session:
        session = sq.ensure_db_session(session_id, db)

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


@router.get("/active")
def get_active_session(db: DBSession = Depends(get_db)):
    """Unity 게임 시작 시 unity_queue 맨 앞을 runtime으로 pop한다."""
    session_id = sq.pop_unity_for_runtime()
    if session_id:
        sq.ensure_db_session(session_id, db)
    return {"session_id": session_id}


@router.get("/unity-queue")
def get_unity_queue():
    """Unity 대기 큐 전체 목록을 순서대로 반환한다 (관리자용)."""
    return {"queue": sq.list_queue(sq.UNITY_QUEUE_KEY)}


@router.post("/unity-queue/advance")
def advance_unity_queue():
    """Unity 체험 완료 시 active 세션을 history_queue로 보낸다."""
    r = sq.redis_conn()
    try:
        active = r.get(sq.ACTIVE_SESSION_KEY)
    finally:
        r.close()
    if active:
        sq.move_active_to_history(active)
    return {"removed": active, "next_session_id": None}


@router.delete("/unity-queue/{session_id}")
def remove_from_unity_queue(session_id: str):
    """대기 큐에서 특정 세션을 제거한다 (관리자용)."""
    sq.remove_from_all_queues(session_id)
    return {"status": "ok", "removed": session_id}


@router.post("/{session_id}/finalize")
def finalize_session(session_id: str, db: DBSession = Depends(get_db)):
    """앱의 이동하기 버튼: session_queue에서 unity_queue로 이동한다."""
    try:
        sq.move_to_unity_queue(session_id)
    except KeyError:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    snapshot = sq.queue_snapshot()
    queue = [item["session_id"] for item in snapshot["unity_queue"]]
    return {
        "status": "ok",
        "session_id": session_id,
        "ready_for_unity": False,
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
    sq.delete_session_state(session_id)
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
