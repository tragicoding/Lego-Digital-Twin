from pydantic import BaseModel
from typing import Literal


class HardwareEvent(BaseModel):
    type: Literal["clap", "light"]
    value: float = 0.0


class CaptureResponse(BaseModel):
    task_id: str
    status: str
    filename: str
