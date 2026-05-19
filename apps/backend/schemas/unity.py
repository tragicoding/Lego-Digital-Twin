from pydantic import BaseModel
from typing import Optional


class AnimationInfo(BaseModel):
    key: str
    display_name: str
    unity_function: str


class CharacterAsset(BaseModel):
    asset_id: str
    model_url: str
    role: str = "guide_npc"
    npc_name: Optional[str] = None
    animations: dict[str, AnimationInfo]


class ObjectAsset(BaseModel):
    asset_id: str
    model_url: str
    role: str = "static_object"
    object_name: Optional[str] = None


class UnitySessionResponse(BaseModel):
    session_id: str
    character_npc_name: Optional[str]   # 캐릭터 NPC 이름
    object_name: Optional[str]          # 오브제 이름
    bubble_text: Optional[str]
    ready_for_unity: bool
    assets: dict  # character / object
