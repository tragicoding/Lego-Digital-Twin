"""
오브제 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 → GLB 저장 → DB 업데이트

GLB 파일명: object_{순서번호}.glb
"""
import asyncio
from pathlib import Path

from redis import Redis

from ...core.config import STORAGE_MODELS, BACKEND_HOST, REDIS_URL
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...services import tripo_service as tripo

ACTIVE_SESSION_KEY = "lego:active_session"


def _is_cancelled(session_id: str) -> bool:
    r = Redis.from_url(REDIS_URL)
    active = r.get(ACTIVE_SESSION_KEY)
    r.close()
    return active is not None and active.decode() != session_id


def _seq_num(db, asset: Asset) -> int:
    count = db.query(Asset).filter(
        Asset.asset_type == asset.asset_type,
        Asset.created_at < asset.created_at,
    ).count()
    return count + 1


def process_object(asset_id: str):
    asyncio.run(_run(asset_id))


async def _run(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return

    cur_session_id = asset.session_id
    session_id = None
    try:
        asset.status = "processing"
        asset.stage = "generating"
        asset.progress = 10
        db.commit()

        seq = _seq_num(db, asset)
        prefix = f"object_{seq}"

        token = await tripo.upload_image(Path(asset.input_image_path))
        if _is_cancelled(cur_session_id):
            return

        task_id = await tripo.create_model_task(token)
        asset.progress = 40
        db.commit()

        result = await tripo.poll_task(task_id)
        if _is_cancelled(cur_session_id):
            return

        asset.stage = "downloading"
        asset.progress = 80
        db.commit()

        glb_path = STORAGE_MODELS / f"{prefix}.glb"
        await tripo.download_glb(result, glb_path)

        asset.model_url = f"{BACKEND_HOST}/static/models/{glb_path.name}"
        asset.status = "completed"
        asset.stage = "ready"
        asset.progress = 100
        db.commit()

        session_id = asset.session_id

    except Exception as e:
        asset.status = "failed"
        asset.error_message = str(e) or repr(e)
        db.commit()
    finally:
        db.close()

    if session_id:
        from ...services.event_service import check_and_notify
        check_and_notify(session_id)
