from pydantic import BaseModel
from typing import Optional


class CharacterAsset(BaseModel):
    asset_id: str
    model_url: str          # 리깅된 FBX URL (Unity Animator 리타겟팅용)
    texture_url: Optional[str] = None  # PBR 텍스쳐 소스 GLB URL (glTFast로 텍스쳐 추출)
    role: str = "guide_npc"
    npc_name: Optional[str] = None


class ObjectAsset(BaseModel):
    asset_id: str
    model_url: str          # GLB URL (glTFast 런타임 로드)
    role: str = "static_object"
    object_name: Optional[str] = None


class UnitySessionResponse(BaseModel):
    session_id: str
    character_npc_name: Optional[str]   # 캐릭터 NPC 이름
    object_name: Optional[str]          # 오브제 이름
    bubble_text: Optional[str]
    ready_for_unity: bool
    assets: dict  # character / object
