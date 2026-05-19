"""
Redis Pub/Sub 이벤트 서비스.

Worker (별도 프로세스) → redis.publish() → FastAPI subscriber → WebSocket broadcast

채널: lego:events
메시지 형식: {"event": "session_ready", "session_id": "s_xxx"}
"""
import json
import asyncio

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from ..core.config import REDIS_URL

CHANNEL = "lego:events"


# ── Worker 쪽 (동기) ────────────────────────────────────────────

def publish_session_ready(session_id: str):
    """Worker 프로세스에서 호출 — 세션 완료 이벤트를 Redis에 발행."""
    r = Redis.from_url(REDIS_URL)
    payload = json.dumps({"event": "session_ready", "session_id": session_id})
    r.publish(CHANNEL, payload)
    r.close()


def check_and_notify(session_id: str):
    """
    세션의 모든 에셋이 completed 상태인지 확인 후 이벤트 발행.
    각 Worker 태스크 완료 시점에 호출한다.
    """
    from ..core.database import SessionLocal
    from ..models.session import Session

    db = SessionLocal()
    try:
        session = db.get(Session, session_id)
        if not session or not session.nickname:
            return
        if not session.assets:
            return
        if all(a.status == "completed" for a in session.assets):
            publish_session_ready(session_id)
    finally:
        db.close()


# ── FastAPI 쪽 (비동기 subscriber) ─────────────────────────────

async def redis_subscriber(broadcast_fn):
    """
    FastAPI lifespan에서 실행되는 영구 subscriber.
    Redis 채널을 구독하고 메시지를 WebSocket으로 브로드캐스트한다.
    """
    r = AsyncRedis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(CHANNEL)

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            payload = json.loads(message["data"])
            await broadcast_fn(payload)
        except Exception:
            pass
