from redis import Redis
from rq import Queue

from ..core.config import REDIS_URL, RQ_QUEUE_CHARACTER, RQ_QUEUE_OBJECT

_redis = Redis.from_url(REDIS_URL)
_queues = {
    "character": Queue(RQ_QUEUE_CHARACTER, connection=_redis),
    "object":    Queue(RQ_QUEUE_OBJECT,    connection=_redis),
}


def enqueue_asset(asset_id: str, asset_type: str):
    """asset_id를 타입별 전용 큐에 등록 — character/object 병렬 처리."""
    if asset_type == "character":
        from ..worker.tasks.character_task import process_character
        _queues["character"].enqueue(process_character, asset_id, job_timeout=600)

    elif asset_type in ("object", "building"):
        from ..worker.tasks.object_task import process_object
        _queues["object"].enqueue(process_object, asset_id, job_timeout=600)
