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
from ..core.config import REDIS_URL, RQ_QUEUE_NAME

#Redis 서버에 연결.
_redis = Redis.from_url(REDIS_URL)
#queue 객체 생성.
_queue = Queue(RQ_QUEUE_NAME, connection=_redis)

#업로드된 에셋을 작업 큐에 등록하는 함수
#인자는 2개
#asset_id    → DB에 저장된 Asset의 고유 ID
#asset_type  → character / building / vehicle
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
