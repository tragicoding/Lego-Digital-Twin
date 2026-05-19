from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SessionCreateResponse(BaseModel):
    session_id: str


class ProfileUpdate(BaseModel):
    nickname: Optional[str] = None
    character_npc_name: Optional[str] = None  # → nickname
    object_name: Optional[str] = None          # → bubble_text
    phone: Optional[str] = None
    bubble_text: Optional[str] = None
    favorite_theme: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    nickname: Optional[str]
    bubble_text: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
