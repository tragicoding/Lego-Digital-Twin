"""
Unity 데이터 라우터
GET /unity/sessions/{session_id} — Unity가 사용할 완성 데이터
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ...core.database import get_db
from ...models.session import Session
from ...schemas.unity import UnitySessionResponse

router = APIRouter(prefix="/unity", tags=["unity"])


@router.get("/sessions/{session_id}", response_model=UnitySessionResponse)
def get_unity_session(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    assets_out = {}
    all_completed = True

    for asset in session.assets:
        if asset.status != "completed" or not asset.model_url:
            all_completed = False
            continue

        if asset.asset_type == "character":
            # model_url  : 리깅된 FBX (Unity Animator 리타겟팅용)
            # texture_url: PBR 텍스쳐 소스 GLB (glTFast로 텍스쳐 추출 후 FBX 캐릭터에 적용)
            assets_out["character"] = {
                "asset_id": asset.id,
                "model_url": asset.model_url,
                "texture_url": asset.thumbnail_url,
                "role": "guide_npc",
                "npc_name": session.nickname,
            }

        elif asset.asset_type in ("object", "building"):
            assets_out["object"] = {
                "asset_id": asset.id,
                "model_url": asset.model_url,
                "role": "static_object",
                "object_name": session.bubble_text,  # bubble_text에 object_name이 저장됨
            }

    ready = all_completed and bool(session.nickname) and len(assets_out) > 0

    return UnitySessionResponse(
        session_id=session_id,
        character_npc_name=session.nickname,
        object_name=session.bubble_text,
        bubble_text=session.bubble_text,
        ready_for_unity=ready,
        assets=assets_out,
    )
