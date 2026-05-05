import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self):
        self._active: set[WebSocket] = set()

    @property
    def count(self) -> int:
        return len(self._active)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket):
        self._active.discard(ws)

    async def broadcast(self, message: dict) -> int:
        dead: set[WebSocket] = set()
        sent = 0
        payload = json.dumps(message, ensure_ascii=False)
        for ws in self._active:
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception:
                dead.add(ws)
        self._active -= dead
        return sent


manager = ConnectionManager()


@router.websocket("/ws/unity")
async def unity_ws(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_text(json.dumps({"type": "connected", "message": "backend 연결 성공"}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
