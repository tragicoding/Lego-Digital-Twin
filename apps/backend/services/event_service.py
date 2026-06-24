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
UNITY_QUEUE_KEY = "lego:unity_queue"


# ── Worker 쪽 (동기) ────────────────────────────────────────────

def publish_session_ready(session_id: str):
    """Worker 프로세스에서 호출 — 세션 완료 이벤트를 Redis에 발행."""
    r = Redis.from_url(REDIS_URL)
    payload = json.dumps({"event": "session_ready", "session_id": session_id})
    r.publish(CHANNEL, payload)
    r.close()


def publish_likes_updated(session_id: str, likes: int, top_session_id: str | None):
    """좋아요 변경 시 호출 — 실시간 업데이트를 모든 Unity 클라이언트에 브로드캐스트."""
    r = Redis.from_url(REDIS_URL)
    payload = json.dumps({
        "event": "likes_updated",
        "session_id": session_id,
        "likes": likes,
        "top_session_id": top_session_id,
    })
    r.publish(CHANNEL, payload)
    r.close()


def enqueue_unity_session(session_id: str) -> bool:
    """Unity 대기 큐에 세션 추가 (중복 방지). 새로 추가됐으면 True."""
    r = Redis.from_url(REDIS_URL)
    try:
        if r.lpos(UNITY_QUEUE_KEY, session_id) is None:
            r.rpush(UNITY_QUEUE_KEY, session_id)
            return True
        return False
    finally:
        r.close()


def check_and_notify(session_id: str):
    """
    세션의 모든 에셋이 completed 상태인지 확인 후 이벤트 발행 + Unity 대기 큐 등록.
    각 Worker 태스크 완료 시점에 호출한다.
    """
    from ..core.database import SessionLocal
    from ..models.session import Session

    db = SessionLocal()
    try:
        session = db.get(Session, session_id)
        if not session:
            return
        if session.status in ("cancelled", "runtime", "completed"):
            return
        if not session.assets:
            return
        completed_types = {a.asset_type for a in session.assets if a.status == "completed"}
        has_character = "character" in completed_types
        has_object = bool({"object", "building"} & completed_types)
        if has_character and has_object:
            enqueued = enqueue_unity_session(session_id)
            if session.status != "unity_queue":
                session.status = "unity_queue"
                db.commit()
            if enqueued:
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
