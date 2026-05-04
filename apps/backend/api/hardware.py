"""
하드웨어 이벤트 REST 엔드포인트.

Arduino 컨트롤러가 이벤트를 POST하면 Unity에 웹소켓으로 전달한다.

지원 이벤트 타입
----------------
clap   — 박수 감지 → Unity 불꽃 이벤트 트리거
light  — 조도 측정 → Unity atmosphere 변경
        lux 값에 따른 atmosphere 매핑:
          >2000  : "day"
          800~2000: "cloudy"
          200~800 : "sunset"
          50~200  : "night"
          <50     : "midnight"

엔드포인트
----------
POST /hardware/event  — Arduino → backend
GET  /hardware/status — 현재 조명 상태 확인용
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

from ..core import state

router = APIRouter(prefix="/hardware", tags=["hardware"])

# 조도(lux) 기준 atmosphere 매핑
LUX_MAP = [
    (2000, "day"),
    (800,  "cloudy"),
    (200,  "sunset"),
    (50,   "night"),
    (0,    "midnight"),
]


def lux_to_atmosphere(lux: float) -> str:
    for threshold, label in LUX_MAP:
        if lux >= threshold:
            return label
    return "midnight"


class HardwareEvent(BaseModel):
    type:  Literal["clap", "light"]
    value: float = 0.0  # clap=클랩 강도 or lux 값


@router.post("/event")
async def receive_hardware_event(event: HardwareEvent):
    """
    Arduino에서 발생한 이벤트를 수신하고 Unity로 중계한다.
    """
    if event.type == "clap":
        msg = {
            "type":      "fireworks",
            "intensity": event.value,
        }

    elif event.type == "light":
        atm = lux_to_atmosphere(event.value)
        async with state.lock:
            state.latest_light = {"lux": event.value, "atmosphere": atm}
        msg = {
            "type":        "atmosphere",
            "atmosphere":  atm,
            "lux":         event.value,
        }

    else:
        return {"status": "unknown event type"}

    sent = await state.broadcast_to_unity(msg)
    return {"status": "ok", "unity_clients": sent, "message": msg}


@router.get("/status")
async def get_light_status():
    """현재 저장된 조명 상태를 반환한다."""
    return state.latest_light
