"""
Unity 데이터 라우터

GET /unity/sessions/{session_id}   — 현재 관람객 세션 (가이드 모드용)
GET /unity/plaza/sessions          — 광장 전체 세션 목록 (자유 모드용)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import desc

from ...core.database import get_db
from ...models.session import Session
from ...models.asset import Asset
from ...models.asset_animation import AssetAnimation
from ...models.plaza_object import PlazaObject
from ...schemas.unity import UnitySessionResponse, PlazaResponse, PlazaSessionData
from ...services import session_queue_service as sq

router = APIRouter(prefix="/unity", tags=["unity"])

MAX_PLAZA_SESSIONS = 31
PROTECTED_SESSIONS_COUNT = 8


def _cleanup_plaza_sessions(db: DBSession, plaza_list: list) -> None:
    """광장 세션 수가 MAX_PLAZA_SESSIONS(31)를 초과하면 오래된 세션을 삭제한다.
    좋아요 상위 PROTECTED_SESSIONS_COUNT(8)개는 삭제 대상에서 제외된다."""
    if len(plaza_list) <= MAX_PLAZA_SESSIONS:
        return

    protected_ids = {
        item["session"].id
        for item in sorted(plaza_list, key=lambda x: x["session"].likes or 0, reverse=True)[:PROTECTED_SESSIONS_COUNT]
    }

    removable = sorted(
        [item for item in plaza_list if item["session"].id not in protected_ids],
        key=lambda x: x["session"].created_at,
    )

    excess = len(plaza_list) - MAX_PLAZA_SESSIONS
    deleted_ids = set()
    for item in removable[:excess]:
        session = item["session"]
        deleted_ids.add(session.id)
        asset_ids = [a.id for a in session.assets]
        if asset_ids:
            db.query(AssetAnimation).filter(AssetAnimation.asset_id.in_(asset_ids)).delete(synchronize_session=False)
        db.query(PlazaObject).filter(PlazaObject.session_id == session.id).delete(synchronize_session=False)
        db.query(Asset).filter(Asset.session_id == session.id).delete(synchronize_session=False)
        db.delete(session)
    db.commit()

    plaza_list[:] = [item for item in plaza_list if item["session"].id not in deleted_ids]


def _build_assets(session: Session) -> dict:
    """Session 모델에서 Unity용 assets dict를 만든다."""
    assets_out = {}
    for asset in session.assets:
        if asset.status != "completed" or not asset.model_url:
            continue
        if asset.asset_type == "character":
            assets_out["character"] = {
                "asset_id": asset.id,
                "model_url": asset.model_url,
                "texture_url": asset.thumbnail_url,   # pbr_model GLB
                "role": "guide_npc",
                "npc_name": session.nickname,
            }
        elif asset.asset_type in ("object", "building"):
            assets_out["object"] = {
                "asset_id": asset.id,
                "model_url": asset.model_url,
                "role": "static_object",
                "object_name": session.bubble_text,
            }
    return assets_out


@router.get("/sessions/{session_id}", response_model=UnitySessionResponse)
def get_unity_session(session_id: str, db: DBSession = Depends(get_db)):
    """현재 관람객 세션 데이터. 가이드 모드 시작 시 Unity가 한 번 호출."""
    session = db.get(Session, session_id)
    if not session and sq.get_session_data(session_id):
        session = sq.ensure_db_session(session_id, db)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    if session.status == "cancelled":
        raise HTTPException(410, "취소된 세션입니다.")

    assets_out = _build_assets(session)
    ready = "character" in assets_out and "object" in assets_out

    return UnitySessionResponse(
        session_id=session_id,
        character_npc_name=session.nickname,
        object_name=session.bubble_text,
        bubble_text=session.bubble_text,
        signature_motion=session.signature_motion,
        likes=session.likes or 0,
        ready_for_unity=ready,
        assets=assets_out,
    )


@router.get("/plaza/sessions", response_model=PlazaResponse)
def get_plaza_sessions(db: DBSession = Depends(get_db)):
    """
    광장에 전시할 이전 관람객들의 세션 목록.
    자유 모드 진입 시 Unity가 호출. 완료된 세션만 반환.
    좋아요 1위 세션에 is_top_liked=True, top_session_id 포함.
    """
    # 완료된 에셋을 가진 세션만
    sessions = db.query(Session).filter(
        Session.status != "cancelled",
        Session.is_true.is_(True),
    ).all()

    plaza_list = []
    for session in sessions:
        assets_out = _build_assets(session)
        if "character" not in assets_out or "object" not in assets_out:
            continue
        plaza_list.append({
            "session": session,
            "assets": assets_out,
        })

    # 31개 초과 시 오래된 비보호 세션 삭제 (상위 8개 보호)
    _cleanup_plaza_sessions(db, plaza_list)

    # 좋아요 1위
    top_session_id = None
    if plaza_list:
        top = max(plaza_list, key=lambda x: x["session"].likes or 0)
        top_session_id = top["session"].id

    result = []
    for item in plaza_list:
        s = item["session"]
        result.append(PlazaSessionData(
            session_id=s.id,
            character_npc_name=s.nickname,
            bubble_text=s.bubble_text,
            signature_motion=s.signature_motion,
            likes=s.likes or 0,
            is_top_liked=(s.id == top_session_id),
            assets=item["assets"],
        ))

    # 좋아요 많은 순 정렬
    result.sort(key=lambda x: x.likes, reverse=True)

    return PlazaResponse(sessions=result, top_session_id=top_session_id)
