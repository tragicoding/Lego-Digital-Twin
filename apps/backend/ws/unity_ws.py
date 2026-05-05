"""
Unity 웹소켓 연결 핸들러.

Unity 클라이언트가 ws://backend:8000/ws/unity 에 연결하면
연결 목록에 추가되고, 연결이 끊기면 자동으로 제거된다.

연결 직후 최신 복셀/조명 데이터를 즉시 전송해
Unity가 씬을 빠르게 초기화할 수 있도록 한다.

메시지 포맷 (JSON)
------------------
서버 → Unity:
  { "type": "voxel_update",  "payload": {...} }
  { "type": "atmosphere",    "atmosphere": "night", "lux": 30.0 }
  { "type": "fireworks",     "intensity": 1.0 }
  { "type": "connected",     "message": "백엔드 연결 성공" }
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core import state

router = APIRouter()


@router.websocket("/ws/unity")
async def unity_websocket(ws: WebSocket):
    await ws.accept()
    state.unity_connections.add(ws)
    print(f"[ws] Unity 클라이언트 연결됨. 현재 {len(state.unity_connections)}개")

    # 연결 직후 초기 상태 전송
    await ws.send_text(json.dumps({
        "type":    "connected",
        "message": "백엔드 연결 성공",
    }))

    if state.latest_voxels:
        await ws.send_text(json.dumps({
            "type":    "voxel_update",
            "payload": state.latest_voxels,
        }))

    if state.latest_light:
        await ws.send_text(json.dumps({
            "type":       "atmosphere",
            "atmosphere": state.latest_light.get("atmosphere", "day"),
            "lux":        state.latest_light.get("lux", 1000),
        }))

    try:
        # Unity → backend 메시지 수신 루프 (현재는 ping/pong 유지용)
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            # 필요 시 Unity 발신 메시지 처리 가능
            print(f"[ws] Unity 메시지 수신: {data}")
    except WebSocketDisconnect:
        pass
    finally:
        state.unity_connections.discard(ws)
        print(f"[ws] Unity 클라이언트 연결 종료. 남은 {len(state.unity_connections)}개")
