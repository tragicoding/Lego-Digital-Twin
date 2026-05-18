"""
캐릭터 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 → 리깅 → 걷기/idle 애니메이션 → GLB 저장 → DB 업데이트

GLB 파일명: npc_{순서번호}_rigged.glb / npc_{순서번호}_walk.glb / npc_{순서번호}_idle.glb
검증된 preset: walk ✅  idle ✅  run ✅  jump ✅
실패 preset (error_code 1004): hello ❌  wave ❌  dance ❌
"""
import asyncio
from pathlib import Path

from ...core.config import STORAGE_MODELS, BACKEND_HOST
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.asset_animation import AssetAnimation
from ...services import tripo_service as tripo

ANIMATIONS = [
    {"key": "walk",  "display_name": "걷기",    "preset": "preset:walk", "unity_function": "animation_walk"},
    {"key": "idle", "display_name": "idle", "preset": "preset:idle", "unity_function": "animation_idle"},
]


def _set_stage(db, asset: Asset, stage: str, progress: int):
    asset.stage = stage
    asset.progress = progress
    db.commit()


def _seq_num(db, asset: Asset) -> int:
    """같은 asset_type 중 생성 순서 번호를 반환한다."""
    count = db.query(Asset).filter(
        Asset.asset_type == asset.asset_type,
        Asset.created_at < asset.created_at,
    ).count()
    return count + 1


def process_character(asset_id: str):
    asyncio.run(_process_character_async(asset_id))


async def _process_character_async(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return

    session_id = None
    try:
        asset.status = "processing"
        _set_stage(db, asset, "generating", 10)

        seq = _seq_num(db, asset)
        prefix = f"npc_{seq}"

        # 1. 이미지 업로드
        token = await tripo.upload_image(Path(asset.input_image_path))

        # 2. 3D 모델 생성
        model_task_id = await tripo.create_model_task(token)
        _set_stage(db, asset, "generating", 30)
        await tripo.poll_task(model_task_id)
        _set_stage(db, asset, "rigging", 50)

        # 3. 리깅
        rig_task_id = await tripo.create_rig_task(model_task_id)
        rig_result = await tripo.poll_task(rig_task_id)

        rig_glb_path = STORAGE_MODELS / f"{prefix}_rigged.glb"
        await tripo.download_glb(rig_result, rig_glb_path)
        final_glb_path = rig_glb_path
        _set_stage(db, asset, "animating", 65)

        # 4. 애니메이션 (걷기 + 인사_01)
        for i, anim in enumerate(ANIMATIONS):
            anim_task_id = await tripo.create_animation_task(rig_task_id, anim["preset"])
            anim_result = await tripo.poll_task(anim_task_id)

            glb_path = STORAGE_MODELS / f"{prefix}_{anim['key']}.glb"
            await tripo.download_glb(anim_result, glb_path)

            db.add(AssetAnimation(
                asset_id=asset_id,
                animation_key=anim["key"],
                display_name=anim["display_name"],
                file_url=f"{BACKEND_HOST}/static/models/{glb_path.name}",
                unity_function=anim["unity_function"],
            ))
            final_glb_path = glb_path
            _set_stage(db, asset, "animating", 65 + (i + 1) * 10)

        db.commit()
        _set_stage(db, asset, "downloading", 90)

        # 5. 완료
        asset.model_url = f"{BACKEND_HOST}/static/models/{final_glb_path.name}"
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
