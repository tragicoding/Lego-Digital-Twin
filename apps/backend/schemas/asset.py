from pydantic import BaseModel
from typing import Optional


class AssetUploadResponse(BaseModel):
    asset_id: str
    status: str
    asset_type: str


class AssetStatusItem(BaseModel):
    status: str
    stage: str
    progress: int
    model_url: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    profile_completed: bool
    assets: dict[str, AssetStatusItem]
    ready_for_unity: bool
