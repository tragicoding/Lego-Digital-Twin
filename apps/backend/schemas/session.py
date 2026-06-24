from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SessionCreateResponse(BaseModel):
    session_id: str


class ProfileUpdate(BaseModel):
    nickname: Optional[str] = None
    character_npc_name: Optional[str] = None  # → nickname
    object_name: Optional[str] = None
    phone: Optional[str] = None
    bubble_text: Optional[str] = None
    favorite_theme: Optional[str] = None


class SignatureMotionUpdate(BaseModel):
    signature_motion: str


class CharacterCompositionUpdate(BaseModel):
    bottom: int
    middle: int
    top: int


class SessionResponse(BaseModel):
    session_id: str
    nickname: Optional[str]
    object_name: Optional[str]
    bubble_text: Optional[str]
    character_bottom: Optional[int] = None
    character_middle: Optional[int] = None
    character_top: Optional[int] = None
    character_number: Optional[int] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
