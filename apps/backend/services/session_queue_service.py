from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any

from redis import Redis
from sqlalchemy.orm import Session as DBSession

from ..core.config import BACKEND_HOST, REDIS_URL, STORAGE_MODELS
from ..models.asset import Asset
from ..models.session import Session

SESSION_QUEUE_KEY = "lego:session_queue"
UNITY_QUEUE_KEY = "lego:unity_queue"
HISTORY_QUEUE_KEY = "lego:history_queue"
ACTIVE_SESSION_KEY = "lego:active_session"
SESSION_SEQ_KEY = "lego:session_seq"
OBJECT_SEQ_KEY = "lego:object_seq"
SESSION_HASH_PREFIX = "lego:session:"


def redis_conn() -> Redis:
    return Redis.from_url(REDIS_URL, decode_responses=True)


def session_key(session_id: str) -> str:
    return f"{SESSION_HASH_PREFIX}{session_id}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def model_url(path: Path) -> str:
    rel = path.relative_to(STORAGE_MODELS).as_posix()
    return f"{BACKEND_HOST}/static/models/{rel}"


def character_files(character_no: int) -> dict[str, str]:
    folder = STORAGE_MODELS / f"Character_{character_no}"
    fbx = folder / f"Character_{character_no}.fbx"
    texture_dir = folder / f"Character_{character_no}_texture"
    if not fbx.exists():
        raise FileNotFoundError(f"Character_{character_no}.fbx not found")

    out = {
        "model_url": model_url(fbx),
        "texture_base_url": model_url(texture_dir) if texture_dir.exists() else "",
        "color_texture_url": "",
        "normal_texture_url": "",
        "metallic_texture_url": "",
        "roughness_texture_url": "",
    }
    if texture_dir.exists():
        candidates = {
            "color_texture_url": ["Color.jpg", "Color.jpeg", "Color.png"],
            "normal_texture_url": ["Normal.jpg", "Normal.jpeg", "Normal.png"],
            "metallic_texture_url": ["legominifigure3dmodel_metallic.JPEG"],
            "roughness_texture_url": ["legominifigure3dmodel_roughness.JPEG"],
        }
        for field, names in candidates.items():
            for name in names:
                path = texture_dir / name
                if path.exists():
                    out[field] = model_url(path)
                    break
    return out


def create_session() -> str:
    r = redis_conn()
    try:
        session_id = f"{r.incr(SESSION_SEQ_KEY) - 1:03d}"
        data = {
            "session_id": session_id,
            "status": "session_queue",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "character_no": "",
            "character_model_url": "",
            "character_texture_base_url": "",
            "character_color_texture_url": "",
            "object_status": "waiting",
            "object_stage": "waiting",
            "object_progress": "0",
            "object_model_url": "",
            "object_error": "",
        }
        r.hset(session_key(session_id), mapping=data)
        if r.lpos(SESSION_QUEUE_KEY, session_id) is None:
            r.rpush(SESSION_QUEUE_KEY, session_id)
        return session_id
    finally:
        r.close()


def get_session_data(session_id: str) -> dict[str, str] | None:
    r = redis_conn()
    try:
        data = r.hgetall(session_key(session_id))
        return data or None
    finally:
        r.close()


def update_session(session_id: str, **fields: Any) -> None:
    r = redis_conn()
    try:
        fields["updated_at"] = now_iso()
        r.hset(session_key(session_id), mapping={k: "" if v is None else str(v) for k, v in fields.items()})
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
        for key in (SESSION_QUEUE_KEY, UNITY_QUEUE_KEY, HISTORY_QUEUE_KEY):
            r.lrem(key, 0, session_id)
        if r.get(ACTIVE_SESSION_KEY) == session_id:
            r.delete(ACTIVE_SESSION_KEY)
    finally:
        r.close()


def delete_session_state(session_id: str) -> None:
    r = redis_conn()
    try:
        for key in (SESSION_QUEUE_KEY, UNITY_QUEUE_KEY, HISTORY_QUEUE_KEY):
            r.lrem(key, 0, session_id)
        if r.get(ACTIVE_SESSION_KEY) == session_id:
            r.delete(ACTIVE_SESSION_KEY)
        r.delete(session_key(session_id))
    finally:
        r.close()


def register_character(session_id: str, character_no: int) -> dict[str, str]:
    files = character_files(character_no)
    r = redis_conn()
    try:
        if not r.exists(session_key(session_id)):
            raise KeyError(session_id)
        r.hset(session_key(session_id), mapping={
            "character_no": str(character_no),
            "character_model_url": files["model_url"],
            "character_texture_base_url": files["texture_base_url"],
            "character_color_texture_url": files["color_texture_url"],
            "updated_at": now_iso(),
        })
        return files
    finally:
        r.close()


def move_to_unity_queue(session_id: str) -> None:
    r = redis_conn()
    try:
        if not r.exists(session_key(session_id)):
            raise KeyError(session_id)
        r.lrem(SESSION_QUEUE_KEY, 0, session_id)
        if r.lpos(UNITY_QUEUE_KEY, session_id) is None:
            r.rpush(UNITY_QUEUE_KEY, session_id)
        r.hset(session_key(session_id), mapping={"status": "unity_queue", "updated_at": now_iso()})
    finally:
        r.close()


def pop_unity_for_runtime() -> str | None:
    r = redis_conn()
    try:
        session_id = r.lpop(UNITY_QUEUE_KEY)
        if not session_id:
            return None
        r.set(ACTIVE_SESSION_KEY, session_id)
        r.hset(session_key(session_id), mapping={"status": "runtime", "updated_at": now_iso()})
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
        r.hset(session_key(session_id), mapping={"status": "history_queue", "updated_at": now_iso()})
    finally:
        r.close()


def next_object_seq() -> int:
    r = redis_conn()
    try:
        return int(r.incr(OBJECT_SEQ_KEY))
    finally:
        r.close()


def queue_item(session_id: str, position: int) -> dict[str, Any]:
    data = get_session_data(session_id) or {}
    return {
        "session_id": session_id,
        "position": position,
        "status": data.get("status", ""),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "character_no": data.get("character_no") or None,
        "character_model_url": data.get("character_model_url") or None,
        "object_status": data.get("object_status", "waiting"),
        "object_stage": data.get("object_stage", "waiting"),
        "object_progress": int(data.get("object_progress") or 0),
        "object_model_url": data.get("object_model_url") or None,
        "object_error": data.get("object_error") or None,
    }


def queue_snapshot() -> dict[str, Any]:
    r = redis_conn()
    try:
        active = r.get(ACTIVE_SESSION_KEY)
    finally:
        r.close()

    return {
        "active_session_id": active,
        "session_queue": [queue_item(sid, i) for i, sid in enumerate(list_queue(SESSION_QUEUE_KEY), start=1)],
        "unity_queue": [queue_item(sid, i) for i, sid in enumerate(list_queue(UNITY_QUEUE_KEY), start=1)],
        "history_queue": [queue_item(sid, i) for i, sid in enumerate(list_queue(HISTORY_QUEUE_KEY), start=1)],
    }


def ensure_db_session(session_id: str, db: DBSession) -> Session:
    session = db.get(Session, session_id)
    data = get_session_data(session_id) or {}
    if not session:
        session = Session(id=session_id, status="runtime")
        db.add(session)

    session.status = "active"

    character_url = data.get("character_model_url") or ""
    if character_url and not any(a.asset_type == "character" for a in session.assets):
        db.add(Asset(
            session_id=session_id,
            asset_type="character",
            input_image_path=None,
            model_url=character_url,
            thumbnail_url=data.get("character_color_texture_url") or None,
            status="completed",
            stage="ready",
            progress=100,
        ))

    object_url = data.get("object_model_url") or ""
    if object_url and not any(a.asset_type == "object" for a in session.assets):
        db.add(Asset(
            session_id=session_id,
            asset_type="object",
            input_image_path=None,
            model_url=object_url,
            status="completed",
            stage="ready",
            progress=100,
        ))

    db.commit()
    db.refresh(session)
    return session


def history_info(session_id: str, db: DBSession) -> dict[str, Any]:
    session = db.get(Session, session_id)
    return {
        "session_id": session_id,
        "character_name": session.nickname if session else None,
        "object_name": session.object_name if session else None,
        "bubble_text": session.bubble_text if session else None,
    }
