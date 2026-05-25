from pydantic import BaseModel
from typing import Optional, List


# ── 현재 세션 (가이드 모드) ─────────────────────────────────────────

class CharacterAsset(BaseModel):
    asset_id: str
    model_url: str           # 리깅된 FBX URL (Unity Animator 리타겟팅용)
    texture_url: Optional[str] = None  # PBR 텍스쳐 소스 GLB (glTFast 텍스쳐 추출)
    role: str = "guide_npc"
    npc_name: Optional[str] = None


class ObjectAsset(BaseModel):
    asset_id: str
    model_url: str           # GLB URL (glTFast 런타임 로드)
    role: str = "static_object"
    object_name: Optional[str] = None


class UnitySessionResponse(BaseModel):
    session_id: str
    character_npc_name: Optional[str]
    object_name: Optional[str]
    bubble_text: Optional[str]
    likes: int = 0
    ready_for_unity: bool
    assets: dict  # character / object


# ── 광장 (자유 모드) ────────────────────────────────────────────────
# GET /unity/plaza/sessions 응답 — 이전 관람객들의 창작물 목록

class PlazaCharacterAsset(BaseModel):
    model_url: str
    texture_url: Optional[str] = None
    npc_name: Optional[str] = None


class PlazaObjectAsset(BaseModel):
    model_url: str
    object_name: Optional[str] = None


class PlazaSessionData(BaseModel):
    session_id: str
    character_npc_name: Optional[str]
    bubble_text: Optional[str]
    likes: int = 0
    is_top_liked: bool = False   # 좋아요 1위 여부 (별 표시용)
    assets: dict                 # character / object


class PlazaResponse(BaseModel):
    sessions: List[PlazaSessionData]
    top_session_id: Optional[str] = None  # 현재 좋아요 1위 세션


# ── 좋아요 ───────────────────────────────────────────────────────────

class LikeResponse(BaseModel):
    session_id: str
    likes: int
    is_top_liked: bool
    top_session_id: Optional[str] = None
