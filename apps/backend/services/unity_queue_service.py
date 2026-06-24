from __future__ import annotations

from redis import Redis

from ..core.config import REDIS_URL

UNITY_QUEUE_KEY = "lego:unity_queue"
HISTORY_QUEUE_KEY = "lego:history_queue"
ACTIVE_SESSION_KEY = "lego:active_session"


def redis_conn() -> Redis:
    return Redis.from_url(REDIS_URL, decode_responses=True)


def get_active_session_id() -> str | None:
    r = redis_conn()
    try:
        return r.get(ACTIVE_SESSION_KEY)
    finally:
        r.close()


def list_queue(key: str) -> list[str]:
    r = redis_conn()
    try:
        return r.lrange(key, 0, -1)
    finally:
        r.close()


def remove_from_all_queues(session_id: str) -> None:
    r = redis_conn()
    try:
        for key in (UNITY_QUEUE_KEY, HISTORY_QUEUE_KEY):
            r.lrem(key, 0, session_id)
        if r.get(ACTIVE_SESSION_KEY) == session_id:
            r.delete(ACTIVE_SESSION_KEY)
    finally:
        r.close()


def pop_unity_for_runtime() -> str | None:
    r = redis_conn()
    try:
        session_id = r.lpop(UNITY_QUEUE_KEY)
        if not session_id:
            return None
        r.set(ACTIVE_SESSION_KEY, session_id)
        return session_id
    finally:
        r.close()


def move_active_to_history(session_id: str) -> None:
    r = redis_conn()
    try:
        if r.get(ACTIVE_SESSION_KEY) == session_id:
            r.delete(ACTIVE_SESSION_KEY)
        if r.lpos(HISTORY_QUEUE_KEY, session_id) is None:
            r.rpush(HISTORY_QUEUE_KEY, session_id)
    finally:
        r.close()
