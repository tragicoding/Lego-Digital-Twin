"""
캐릭터 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 → 리깅 → 걷기/인사_01 애니메이션 → GLB 저장 → DB 업데이트
"""
import asyncio
from pathlib import Path

from ...core.config import STORAGE_MODELS, BACKEND_HOST
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.asset_animation import AssetAnimation
from ...services import tripo_service as tripo

# Tripo 애니메이션 preset 매핑
ANIMATIONS = [
    {"key": "walk",  "display_name": "걷기",    "preset": "preset:walk",  "unity_function": "animation_walk"},
    {"key": "hello", "display_name": "인사_01", "preset": "preset:wave",  "unity_function": "animation_Hello"},
]


def _set_stage(db, asset: Asset, stage: str, progress: int):
    asset.stage = stage
    asset.progress = progress
    db.commit()


def process_character(asset_id: str):
    """RQ Worker 진입점 — 동기 래퍼."""
    asyncio.run(_process_character_async(asset_id))


async def _process_character_async(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return

    try:
        asset.status = "processing"
        _set_stage(db, asset, "generating", 10)

        # 1. 이미지 업로드
        token = await tripo.upload_image(Path(asset.input_image_path))

        # 2. 3D 모델 생성
        model_task_id = await tripo.create_model_task(token)
        _set_stage(db, asset, "generating", 30)
        model_result = await tripo.poll_task(model_task_id)
        _set_stage(db, asset, "rigging", 50)

        # 3. 리깅
        rig_task_id = await tripo.create_rig_task(model_task_id)
        rig_result = await tripo.poll_task(rig_task_id)
        _set_stage(db, asset, "animating", 65)

        # 4. 애니메이션 (걷기 + 인사_01) — 기본은 마지막 애니메이션 GLB 사용
        final_glb_path = None
        for i, anim in enumerate(ANIMATIONS):
            anim_task_id = await tripo.create_animation_task(rig_task_id, anim["preset"])
            anim_result = await tripo.poll_task(anim_task_id)

            glb_path = STORAGE_MODELS / f"{asset_id}_{anim['key']}.glb"
            await tripo.download_glb(anim_result, glb_path)

            # animation 메타데이터 DB 저장
            db_anim = AssetAnimation(
                asset_id=asset_id,
                animation_key=anim["key"],
                display_name=anim["display_name"],
                file_url=f"{BACKEND_HOST}/static/models/{glb_path.name}",
                unity_function=anim["unity_function"],
            )
            db.add(db_anim)
            final_glb_path = glb_path
            _set_stage(db, asset, "animating", 65 + (i + 1) * 10)

        db.commit()
        _set_stage(db, asset, "downloading", 90)

        # 5. 최종 GLB (마지막 애니메이션 포함 파일 사용)
        asset.model_url = f"{BACKEND_HOST}/static/models/{final_glb_path.name}"
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
