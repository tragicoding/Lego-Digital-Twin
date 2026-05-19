"""
RQ Worker 시작 스크립트 — character / object 큐 2개 병렬 실행.
프로젝트 루트에서 실행: conda run -n triposr python run_worker.py
"""
import sys
import os
import signal
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(__file__))

from redis import Redis
from rq import Worker, Queue

from apps.backend.core.config import REDIS_URL, RQ_QUEUE_CHARACTER, RQ_QUEUE_OBJECT

QUEUES = [RQ_QUEUE_CHARACTER, RQ_QUEUE_OBJECT]


def run_worker(queue_name: str):
    """각 프로세스에서 실행되는 Worker — 전용 큐 하나만 담당."""
    conn = Redis.from_url(REDIS_URL)
    q = Queue(queue_name, connection=conn)
    w = Worker([q], connection=conn)
    print(f"[Worker] 시작 — queue: {queue_name}", flush=True)
    w.work()


if __name__ == "__main__":
    processes: list[Process] = []

    for queue_name in QUEUES:
        p = Process(target=run_worker, args=(queue_name,), name=f"worker-{queue_name}")
        p.start()
        processes.append(p)

    print(f"[WorkerPool] {len(QUEUES)}개 Worker 병렬 실행 중: {QUEUES}", flush=True)

    def _shutdown(sig, frame):
        print("\n[WorkerPool] 종료 신호 수신 — Worker 정리 중...", flush=True)
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for p in processes:
        p.join()
