"""
Python + Redis + RQ를 사용해서, 업로드된 에셋을 백그라운드 작업 큐에 등록
이미지 업로드 후 3D 변환 작업을 worker에게 넘기는 역할

CapturePage에서 이미지 업로드
↓
POST /sessions/{session_id}/assets
↓
서버가 이미지 저장
↓
DB에 Asset 생성
↓
enqueue_asset(asset.id, asset_type)
↓
asset_type에 따라 작업 함수 선택
↓
Redis Queue에 작업 등록
↓
RQ Worker가 작업 실행
↓
process_character / process_building / process_vehicle 실행
"""


"""
Redis: Redis 서버에 연결하기 위한 라이브러리
Queue: RQ에서 작업 큐를 만들기 위한 클래스"""
from redis import Redis
from rq import Queue

#설정파일에서 Redis 주소와 Queue 이름을 가져온다.
'''
예상값
REDIS_URL = "redis://localhost:6379/0"
RQ_QUEUE_NAME = "lego_tasks"
'''
from ..core.config import REDIS_URL, RQ_QUEUE_CHARACTER, RQ_QUEUE_BUILDING, RQ_QUEUE_VEHICLE

_redis = Redis.from_url(REDIS_URL)
_queues = {
    "character": Queue(RQ_QUEUE_CHARACTER, connection=_redis),
    "building":  Queue(RQ_QUEUE_BUILDING,  connection=_redis),
    "vehicle":   Queue(RQ_QUEUE_VEHICLE,   connection=_redis),
}


def enqueue_asset(asset_id: str, asset_type: str):
    """asset_id를 타입별 전용 큐에 등록 — 3개 큐가 병렬로 처리된다."""
    if asset_type == "character":
        from ..worker.tasks.character_task import process_character
        _queues["character"].enqueue(process_character, asset_id, job_timeout=600)

    elif asset_type == "building":
        from ..worker.tasks.building_task import process_building
        _queues["building"].enqueue(process_building, asset_id, job_timeout=600)

    elif asset_type == "vehicle":
        from ..worker.tasks.vehicle_task import process_vehicle
        _queues["vehicle"].enqueue(process_vehicle, asset_id, job_timeout=600)


def enqueue_mock_asset(asset_id: str, asset_type: str):
    """Mock 파이프라인용 — 타입별 큐에 등록."""
    if asset_type == "character":
        from ..worker.tasks.mock_task import process_mock_character
        _queues["character"].enqueue(process_mock_character, asset_id, job_timeout=120)

    elif asset_type == "building":
        from ..worker.tasks.mock_task import process_mock_building
        _queues["building"].enqueue(process_mock_building, asset_id, job_timeout=120)

    elif asset_type == "vehicle":
        from ..worker.tasks.mock_task import process_mock_vehicle
        _queues["vehicle"].enqueue(process_mock_vehicle, asset_id, job_timeout=120)
