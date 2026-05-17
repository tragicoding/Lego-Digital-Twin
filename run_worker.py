"""
RQ Worker 시작 스크립트.
프로젝트 루트에서 실행: conda run -n triposr python run_worker.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from redis import Redis
from rq import Worker, Queue

from apps.backend.core.config import REDIS_URL, RQ_QUEUE_NAME

if __name__ == "__main__":
    conn = Redis.from_url(REDIS_URL)
    q = Queue(RQ_QUEUE_NAME, connection=conn)
    w = Worker([q], connection=conn)
    print(f"[Worker] 시작 — queue: {RQ_QUEUE_NAME}, redis: {REDIS_URL}")
    w.work()
