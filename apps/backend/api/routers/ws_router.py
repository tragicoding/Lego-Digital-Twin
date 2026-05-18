"""
Unity WebSocket 엔드포인트.
ws://localhost:8000/ws/unity

연결 후 수신 이벤트:
  {"event": "session_ready", "session_id": "s_xxx"}
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...services.ws_manager import manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/unity")
async def unity_ws(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # 연결 유지 (ping 등 수신 가능)
    except WebSocketDisconnect:
        manager.disconnect(ws)
