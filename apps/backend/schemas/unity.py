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


class BuildingAsset(BaseModel):
    asset_id: str
    model_url: str
    role: str = "static_building"
    owner_name: Optional[str] = None


class VehicleAsset(BaseModel):
    asset_id: str
    model_url: str
    role: str = "driveable_vehicle"
    owner_name: Optional[str] = None
    control_mode: str = "auto_drive"


class UnitySessionResponse(BaseModel):
    session_id: str
    nickname: Optional[str]
    bubble_text: Optional[str]
    ready_for_unity: bool
    assets: dict  # character / building / vehicle
