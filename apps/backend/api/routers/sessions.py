"""
세션 라우터
POST /sessions              — 세션 생성 (관객 입장 즉시)
PATCH /sessions/{id}/profile — 닉네임/말풍선 입력
GET  /sessions/{id}         — 세션 정보 조회
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ...core.database import get_db
from ...models.session import Session
from ...schemas.session import SessionCreateResponse, ProfileUpdate, SessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionCreateResponse, status_code=201)
def create_session(db: DBSession = Depends(get_db)):
    session = Session()
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

    session.nickname = body.nickname
    if body.phone:
        session.phone = body.phone
    if body.bubble_text:
        session.bubble_text = body.bubble_text
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
