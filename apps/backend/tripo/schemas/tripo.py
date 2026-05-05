from pydantic import BaseModel
from typing import Literal, Optional


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["queued", "processing", "success", "failed"]
    filename: Optional[str] = None
    download_url: Optional[str] = None
    unity_path: Optional[str] = None
    error: Optional[str] = None


class ModelListItem(BaseModel):
    filename: str
    download_url: str
    unity_path: str
    size_bytes: int
