import socket
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue, Worker
from rq.registry import (
    DeferredJobRegistry,
    FailedJobRegistry,
    FinishedJobRegistry,
    ScheduledJobRegistry,
    StartedJobRegistry,
)
from sqlalchemy import desc
from sqlalchemy.orm import Session as DBSession

from ...core.config import (
    BACKEND_HOST,
    REDIS_URL,
    RQ_QUEUE_CHARACTER,
    RQ_QUEUE_OBJECT,
    STORAGE_IMAGES,
    STORAGE_MODELS,
)
from ...core.database import get_db
from ...models.asset import Asset
from ...models.asset_animation import AssetAnimation
from ...models.plaza_object import PlazaObject
from ...models.session import Session
from ...services.event_service import check_and_notify
from ...services import unity_queue_service as uq

router = APIRouter(prefix="/admin", tags=["admin"])


ROOT_DIR = Path(__file__).resolve().parents[4]
FRONTEND_ENV = ROOT_DIR / "apps" / "frontend" / ".env.local"
REGISTRY_TYPES = [
    StartedJobRegistry,
    FailedJobRegistry,
    DeferredJobRegistry,
    ScheduledJobRegistry,
    FinishedJobRegistry,
]
MANAGED_QUEUES = {
    "lego-character": RQ_QUEUE_CHARACTER,
    "lego-object": RQ_QUEUE_OBJECT,
}


class CharacterCompositionBody(BaseModel):
    bottom: int
    middle: int
    top: int


def _character_number(bottom: int, middle: int, top: int) -> int:
    return (bottom - 1) * 9 + (middle - 1) * 3 + top


def _validate_part(value: int, name: str) -> None:
    if value not in (1, 2, 3):
        raise HTTPException(422, f"{name} 값은 1, 2, 3 중 하나여야 합니다.")


def _redis() -> Redis:
    return Redis.from_url(REDIS_URL)


def _parse_env_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def _extract_host(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.hostname


def _extract_port(url: str | None, default: int) -> int:
    if not url:
        return default
    parsed = urlparse(url)
    return parsed.port or default


def _probe_port(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get_wsl_ips() -> list[str]:
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True).strip()
    except Exception:
        return []
    return [part for part in output.split() if part]


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _decode_redis_value(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _front_file_exists(session_id: str, asset_prefix: str) -> bool:
    img_dir = STORAGE_IMAGES / session_id
    if not img_dir.exists():
        return False
    legacy_excluded = (f"{asset_prefix}_left_", f"{asset_prefix}_back_", f"{asset_prefix}_right_")
    for path in img_dir.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if stem.endswith("_front") and stem.startswith(f"{asset_prefix}_"):
            return True
        if not path.name.startswith(f"{asset_prefix}_"):
            continue
        if path.name.startswith(legacy_excluded):
            continue
        if any(stem.endswith(f"_{view}") for view in ("left", "back", "right")):
            continue
        return True
    return False


def _has_view_file(session_id: str, asset_prefix: str, view: str) -> bool:
    img_dir = STORAGE_IMAGES / session_id
    if not img_dir.exists():
        return False
    legacy = list(img_dir.glob(f"{asset_prefix}_{view}_*"))
    captured = list(img_dir.glob(f"{asset_prefix}_*_{view}.*"))
    return bool(legacy or captured)


def _view_checks(session_id: str, asset_prefix: str) -> dict:
    left = _has_view_file(session_id, asset_prefix, "left")
    back = _has_view_file(session_id, asset_prefix, "back")
    right = _has_view_file(session_id, asset_prefix, "right")
    front = _front_file_exists(session_id, asset_prefix)
    return {
        "front": front,
        "left": left,
        "back": back,
        "right": right,
        "all_present": front and left and back and right,
    }


def _file_exists_from_url(url: str | None) -> bool:
    if not url:
        return False
    filename = Path(urlparse(url).path).name
    if not filename:
        return False
    return (STORAGE_MODELS / filename).exists()


def _asset_by_type(session: Session, *asset_types: str) -> Asset | None:
    candidates = [
        asset
        for asset in session.assets
        if asset.asset_type in asset_types and asset.status != "superseded"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda asset: (asset.created_at is not None, asset.created_at, asset.id))


def _session_summary(session: Session, queue_position: int | None = None) -> dict:
    character_asset = _asset_by_type(session, "character")
    object_asset = _asset_by_type(session, "object", "building")

    return {
        "session_id": session.id,
        "status": session.status,
        "created_at": _iso(session.created_at),
        "updated_at": _iso(session.updated_at),
        "character_name": session.nickname,
        "object_name": session.object_name,
        "bubble_text": session.bubble_text,
        "signature_motion": session.signature_motion,
        "character_bottom": session.character_bottom,
        "character_middle": session.character_middle,
        "character_top": session.character_top,
        "character_number": session.character_number,
        "character_selection_ready": session.character_number is not None,
        "likes": session.likes or 0,
        "queue_position": queue_position,
        "image_checks": {
            "character": _view_checks(session.id, "character"),
            "object": _view_checks(session.id, "object"),
        },
        "downloads": {
            "character_fbx": _file_exists_from_url(character_asset.model_url if character_asset else None),
            "character_texture_glb": _file_exists_from_url(character_asset.thumbnail_url if character_asset else None),
            "object_glb": _file_exists_from_url(object_asset.model_url if object_asset else None),
        },
        "assets": [
            {
                "asset_id": asset.id,
                "asset_type": asset.asset_type,
                "status": asset.status,
                "stage": asset.stage,
                "progress": asset.progress,
                "model_url": asset.model_url,
                "thumbnail_url": asset.thumbnail_url,
                "created_at": _iso(asset.created_at),
            }
            for asset in session.assets
        ],
    }


def _serialize_job(job) -> dict:
    args = list(job.args) if job.args else []
    return {
        "job_id": job.id,
        "status": job.get_status(refresh=False),
        "func_name": job.func_name,
        "args": args,
        "created_at": _iso(job.created_at),
        "enqueued_at": _iso(job.enqueued_at),
        "ended_at": _iso(job.ended_at),
        "description": job.description,
    }


def _queue_snapshot(name: str, redis_conn: Redis) -> dict:
    queue = Queue(name, connection=redis_conn)
    registries = {
        "queued": [_serialize_job(job) for job in queue.get_jobs()],
    }
    for registry_cls in REGISTRY_TYPES:
        registry = registry_cls(name, connection=redis_conn)
        jobs = []
        for job_id in registry.get_job_ids():
            job = queue.fetch_job(job_id)
            if job is not None:
                jobs.append(_serialize_job(job))
        registries[registry_cls.__name__] = jobs
    return {
        "name": name,
        "count": queue.count,
        "registries": registries,
    }


def _remove_session_everywhere(session_id: str, db: DBSession) -> None:
    exists = db.query(Session.id).filter(Session.id == session_id).first()
    if not exists:
        uq.remove_from_all_queues(session_id)
        return

    asset_ids = {
        asset_id
        for (asset_id,) in db.query(Asset.id).filter(Asset.session_id == session_id).all()
    }

    if asset_ids:
        db.query(AssetAnimation).filter(AssetAnimation.asset_id.in_(asset_ids)).delete(synchronize_session=False)
    db.query(PlazaObject).filter(PlazaObject.session_id == session_id).delete(synchronize_session=False)
    db.query(Asset).filter(Asset.session_id == session_id).delete(synchronize_session=False)
    db.query(Session).filter(Session.id == session_id).delete(synchronize_session=False)
    db.commit()

    redis_conn = _redis()
    try:
        uq.remove_from_all_queues(session_id)
        for queue_name in MANAGED_QUEUES.values():
            queue = Queue(queue_name, connection=redis_conn)
            for job in queue.get_jobs():
                if any(str(arg) in asset_ids for arg in job.args):
                    job.delete(remove_from_queue=True)
            for registry_cls in REGISTRY_TYPES:
                registry = registry_cls(queue_name, connection=redis_conn)
                for job in registry.get_jobs():
                    if any(str(arg) in asset_ids for arg in job.args):
                        registry.remove(job.id, delete_job=True)
    finally:
        redis_conn.close()


def _clear_queue(queue_name: str) -> dict:
    if queue_name not in MANAGED_QUEUES:
        raise HTTPException(404, "알 수 없는 큐입니다.")

    redis_conn = _redis()
    try:
        resolved = MANAGED_QUEUES[queue_name]
        queue = Queue(resolved, connection=redis_conn)
        queue.empty()
        cleared = 0
        for registry_cls in REGISTRY_TYPES:
            registry = registry_cls(resolved, connection=redis_conn)
            for job_id in registry.get_job_ids():
                registry.remove(job_id, delete_job=True)
                cleared += 1
        return {"status": "ok", "queue": resolved, "cleared_registry_jobs": cleared}
    finally:
        redis_conn.close()


@router.get("/dashboard")
def get_admin_dashboard(db: DBSession = Depends(get_db)):
    frontend_api_url = _parse_env_value(FRONTEND_ENV, "VITE_API_URL")
    configured_windows_ip = _extract_host(BACKEND_HOST) or _extract_host(frontend_api_url)
    backend_port = _extract_port(BACKEND_HOST, 8000)
    frontend_port = 3000
    admin_port = 3001

    redis_conn = _redis()
    try:
        redis_ok = bool(redis_conn.ping())
        unity_queue_ids = [_decode_redis_value(item) for item in redis_conn.lrange(uq.UNITY_QUEUE_KEY, 0, -1)]
        history_queue_ids = [_decode_redis_value(item) for item in redis_conn.lrange(uq.HISTORY_QUEUE_KEY, 0, -1)]
        active_session_id = _decode_redis_value(redis_conn.get(uq.ACTIVE_SESSION_KEY))
        worker_list = []
        for worker in Worker.all(connection=redis_conn):
            worker_list.append(
                {
                    "name": worker.name,
                    "state": worker.get_state(),
                    "queues": list(worker.queue_names()),
                    "current_job_id": worker.get_current_job_id(),
                    "last_heartbeat": _iso(worker.last_heartbeat),
                }
            )
        queue_snapshots = [
            _queue_snapshot(queue_name, redis_conn)
            for queue_name in MANAGED_QUEUES.values()
        ]
        redis_keys = {
            "rq": len(redis_conn.keys("rq:*")),
            "lego": len(redis_conn.keys("lego:*")),
        }
    finally:
        redis_conn.close()

    unity_sessions = []
    for index, session_id in enumerate(unity_queue_ids, start=1):
        session = db.get(Session, session_id)
        if session:
            unity_sessions.append(_session_summary(session, queue_position=index))

    history_sessions = []
    for index, session_id in enumerate(history_queue_ids, start=1):
        session = db.get(Session, session_id)
        if session:
            history_sessions.append(_session_summary(session, queue_position=index))

    sessions = db.query(Session).order_by(desc(Session.created_at)).all()
    queued_or_done_ids = set(unity_queue_ids) | set(history_queue_ids)
    if active_session_id:
        queued_or_done_ids.add(active_session_id)
    session_queue = [
        _session_summary(session)
        for session in sessions
        if session.id not in queued_or_done_ids
        and session.status not in ("cancelled", "unity_queue", "runtime", "completed")
    ]
    session_summaries = [_session_summary(session) for session in sessions]

    likes_ranking = [
        {
            "session_id": session.id,
            "character_name": session.nickname,
            "object_name": session.object_name,
            "bubble_text": session.bubble_text,
            "likes": session.likes or 0,
        }
        for session in db.query(Session).order_by(desc(Session.likes), desc(Session.created_at)).limit(20).all()
    ]

    model_files = []
    for path in sorted(STORAGE_MODELS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        stat = path.stat()
        model_files.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    model_files = model_files[:200]

    return {
        "server": {
            "health": "ok",
            "backend_host": BACKEND_HOST,
            "frontend_api_url": frontend_api_url,
            "configured_windows_ip": configured_windows_ip,
            "wsl_ips": _get_wsl_ips(),
            "ports": {
                "localhost": {
                    "backend_8000": _probe_port("127.0.0.1", backend_port),
                    "frontend_3000": _probe_port("127.0.0.1", frontend_port),
                    "admin_3001": _probe_port("127.0.0.1", admin_port),
                    "redis_6379": _probe_port("127.0.0.1", 6379),
                },
                "configured_windows_ip": {
                    "backend_8000": _probe_port(configured_windows_ip, backend_port) if configured_windows_ip else False,
                    "frontend_3000": _probe_port(configured_windows_ip, frontend_port) if configured_windows_ip else False,
                    "admin_3001": _probe_port(configured_windows_ip, admin_port) if configured_windows_ip else False,
                },
            },
        },
        "workers": {
            "workers": worker_list,
            "queues": queue_snapshots,
        },
        "redis": {
            "healthy": redis_ok,
            "key_counts": redis_keys,
            "session_queue_length": len(session_queue),
            "unity_queue_length": len(unity_queue_ids),
            "history_queue_length": len(history_queue_ids),
        },
        "data": {
            "counts": {
                "sessions": db.query(Session).count(),
                "assets": db.query(Asset).count(),
                "plaza_objects": db.query(PlazaObject).count(),
                "asset_animations": db.query(AssetAnimation).count(),
                "models": len(model_files),
            },
            "models": model_files,
            "sessions": session_summaries,
        },
        "exhibition": {
            "current_session_id": active_session_id or (unity_queue_ids[0] if unity_queue_ids else None),
            "active_session_id": active_session_id,
            "session_queue": session_queue,
            "unity_queue": unity_sessions,
            "history_queue": history_sessions,
            "likes_ranking": likes_ranking,
        },
    }


@router.delete("/queues/{queue_name}")
def clear_queue(queue_name: str):
    return _clear_queue(queue_name)


@router.delete("/unity-queue")
def clear_unity_queue():
    redis_conn = _redis()
    try:
        removed = redis_conn.llen(uq.UNITY_QUEUE_KEY)
        redis_conn.delete(uq.UNITY_QUEUE_KEY)
        return {"status": "ok", "removed_count": removed}
    finally:
        redis_conn.close()


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: DBSession = Depends(get_db)):
    _remove_session_everywhere(session_id, db)
    return {"status": "ok", "removed": session_id}


@router.patch("/sessions/{session_id}/character-composition")
def update_character_composition(
    session_id: str,
    body: CharacterCompositionBody,
    db: DBSession = Depends(get_db),
):
    """관리자 페이지에서 하체/상체/머리 조합을 확정한다."""
    _validate_part(body.bottom, "bottom")
    _validate_part(body.middle, "middle")
    _validate_part(body.top, "top")

    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    if session.status == "cancelled":
        raise HTTPException(409, "취소된 세션입니다.")

    character_asset = _asset_by_type(session, "character")
    if character_asset is None:
        character_asset = Asset(
            session_id=session_id,
            asset_type="character",
            status="completed",
            stage="ready",
            progress=100,
        )
        db.add(character_asset)

    session.character_bottom = body.bottom
    session.character_middle = body.middle
    session.character_top = body.top
    session.character_number = _character_number(body.bottom, body.middle, body.top)

    character_asset.status = "completed"
    character_asset.stage = "ready"
    character_asset.progress = 100
    db.commit()

    check_and_notify(session_id)

    return {
        "status": "ok",
        "session_id": session_id,
        "bottom": body.bottom,
        "middle": body.middle,
        "top": body.top,
        "character_number": session.character_number,
    }


@router.delete("/db/reset")
def reset_database(db: DBSession = Depends(get_db)):
    session_ids = [session_id for (session_id,) in db.query(Session.id).all()]
    for session_id in session_ids:
        _remove_session_everywhere(session_id, db)

    redis_conn = _redis()
    try:
        redis_conn.delete(
            uq.UNITY_QUEUE_KEY,
            uq.HISTORY_QUEUE_KEY,
            uq.ACTIVE_SESSION_KEY,
        )
    finally:
        redis_conn.close()

    return {"status": "ok", "removed_sessions": len(session_ids)}
