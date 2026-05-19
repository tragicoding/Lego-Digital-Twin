from redis import Redis
from rq import Queue
from ..core.config import REDIS_URL, RQ_QUEUE_NAME

_redis = Redis.from_url(REDIS_URL)
_queue = Queue(RQ_QUEUE_NAME, connection=_redis)


def enqueue_asset(asset_id: str, asset_type: str):
    """asset_id를 Redis Queue에 등록한다."""
    if asset_type == "character":
        from ..worker.tasks.character_task import process_character
        _queue.enqueue(process_character, asset_id, job_timeout=600)

    elif asset_type == "building":
        from ..worker.tasks.building_task import process_building
        _queue.enqueue(process_building, asset_id, job_timeout=600)

    elif asset_type == "vehicle":
        from ..worker.tasks.vehicle_task import process_vehicle
        _queue.enqueue(process_vehicle, asset_id, job_timeout=600)
