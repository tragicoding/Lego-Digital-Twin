"""건축물 처리 — 3D 생성만 수행 (리깅 없음)"""
import asyncio
from pathlib import Path

from ...core.config import STORAGE_MODELS, BACKEND_HOST
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...services import tripo_service as tripo


def process_building(asset_id: str):
    asyncio.run(_run(asset_id))


async def _run(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return
    try:
        asset.status = "processing"
        asset.stage = "generating"
        asset.progress = 10
        db.commit()

        token = await tripo.upload_image(Path(asset.input_image_path))
        task_id = await tripo.create_model_task(token)

        asset.progress = 40
        db.commit()

        result = await tripo.poll_task(task_id)

        asset.stage = "downloading"
        asset.progress = 80
        db.commit()

        glb_path = STORAGE_MODELS / f"{asset_id}_building.glb"
        await tripo.download_glb(result, glb_path)

        asset.model_url = f"{BACKEND_HOST}/static/models/{glb_path.name}"
        asset.status = "completed"
        asset.stage = "ready"
        asset.progress = 100
        db.commit()

    except Exception as e:
        asset.status = "failed"
        asset.error_message = str(e)
        db.commit()
    finally:
        db.close()
