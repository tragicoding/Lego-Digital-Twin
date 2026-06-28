"""
RQ Worker 시작 스크립트 — character / object 큐 2개 병렬 실행.
프로젝트 루트에서 실행: conda run -n triposr python run_worker.py

워커가 죽으면 자동으로 재시작한다 (2초 대기).
"""
import sys
import os
import signal
import time
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(__file__))

from redis import Redis
from rq import Worker, Queue

from apps.backend.core.config import REDIS_URL, RQ_QUEUE_CHARACTER, RQ_QUEUE_OBJECT

QUEUES = [RQ_QUEUE_CHARACTER, RQ_QUEUE_OBJECT]
RESTART_DELAY = 2  # 재시작 전 대기 시간 (초)

_shutdown_requested = False


def run_worker(queue_name: str):
    """각 프로세스에서 실행되는 Worker — 전용 큐 하나만 담당."""
    conn = Redis.from_url(REDIS_URL)
    q = Queue(queue_name, connection=conn)
    w = Worker([q], connection=conn)
    print(f"[Worker] 시작 — queue: {queue_name}", flush=True)
    w.work()


def supervised_worker(queue_name: str):
    """워커가 죽으면 자동 재시작하는 감시 루프."""
    while not _shutdown_requested:
        p = Process(target=run_worker, args=(queue_name,), name=f"worker-{queue_name}")
        p.start()
        p.join()  # 워커가 죽을 때까지 대기

        if _shutdown_requested:
            break

        exit_code = p.exitcode
        print(f"[Supervisor] {queue_name} 워커 종료 (exit={exit_code}) — {RESTART_DELAY}초 후 재시작", flush=True)
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    supervisor_processes: list[Process] = []

    for queue_name in QUEUES:
        p = Process(target=supervised_worker, args=(queue_name,), name=f"supervisor-{queue_name}")
        p.start()
        supervisor_processes.append(p)

    print(f"[WorkerPool] {len(QUEUES)}개 Worker 병렬 실행 중 (자동 재시작 활성): {QUEUES}", flush=True)

    def _shutdown(sig, frame):
        global _shutdown_requested
        _shutdown_requested = True
        print("\n[WorkerPool] 종료 신호 수신 — Worker 정리 중...", flush=True)
        for p in supervisor_processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for p in supervisor_processes:
        p.join()
