"""
세션 라우터
POST /sessions              — 세션 생성 (관객 입장 즉시)
PATCH /sessions/{id}/profile — 닉네임/말풍선 입력
GET  /sessions/{id}         — 세션 정보 조회
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from redis import Redis

from ...core.config import REDIS_URL, RQ_QUEUE_CHARACTER, RQ_QUEUE_OBJECT
from ...core.database import get_db
from ...models.session import Session
from ...schemas.session import SessionCreateResponse, ProfileUpdate, SessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])

ACTIVE_SESSION_KEY = "lego:active_session"


def _reset_for_new_session(session_id: str):
    """새 세션 시작 시 기존 큐를 비우고 active session을 갱신한다."""
    r = Redis.from_url(REDIS_URL)
    r.delete(f"rq:queue:{RQ_QUEUE_CHARACTER}")
    r.delete(f"rq:queue:{RQ_QUEUE_OBJECT}")
    r.set(ACTIVE_SESSION_KEY, session_id)
    r.close()


@router.post("", response_model=SessionCreateResponse, status_code=201)
def create_session(db: DBSession = Depends(get_db)):
    session = Session()
    db.add(session)
    db.commit()
    db.refresh(session)
    _reset_for_new_session(session.id)
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
    return {"status": "ok", "session_id": session_id}


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    return session
