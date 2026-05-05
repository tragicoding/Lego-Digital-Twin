"""
전역 공유 상태 모듈.

백엔드 서버 안에서 모든 라우터/웹소켓 핸들러가 공유하는 데이터를 보관한다.
멀티스레드 접근에 대비해 asyncio.Lock을 사용한다.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

# 현재 처리된 최신 복셀 페이로드 (engine-vision 이 올린 것)
latest_voxels: dict | None = None

# 현재 조명 상태 (hardware 가 올린 것)
latest_light: dict = {"lux": 1000, "atmosphere": "day"}

# 활성 Unity 웹소켓 연결 목록
unity_connections: set = set()

# 공유 데이터 보호용 락
lock = asyncio.Lock()


async def broadcast_to_unity(message: dict) -> int:
    """
    연결된 모든 Unity 클라이언트에 JSON 메시지를 브로드캐스트한다.
    전송 성공한 클라이언트 수를 반환한다.
    """
    import json
    dead = set()
    sent = 0
    payload_str = json.dumps(message)

    for ws in unity_connections:
        try:
            await ws.send_text(payload_str)
            sent += 1
        except Exception:
            dead.add(ws)

    # 끊어진 연결 정리
    unity_connections.difference_update(dead)
    return sent
