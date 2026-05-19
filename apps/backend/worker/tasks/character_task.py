"""
캐릭터 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 → 리깅 → 걷기/idle 애니메이션 → GLB 저장 → DB 업데이트

GLB 파일명: npc_{순서번호}_rigged.glb / npc_{순서번호}_walk.glb / npc_{순서번호}_idle.glb
검증된 preset: walk ✅  idle ✅  run ✅  jump ✅
실패 preset (error_code 1004): hello ❌  wave ❌  dance ❌
"""
import asyncio
import time
from pathlib import Path

from redis import Redis

from ...core.config import STORAGE_MODELS, BACKEND_HOST, REDIS_URL
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.asset_animation import AssetAnimation
from ...services import tripo_service as tripo

ACTIVE_SESSION_KEY = "lego:active_session"


def _is_cancelled(session_id: str) -> bool:
    """현재 세션이 active session이 아니면 True (새 세션이 시작된 것)."""
    r = Redis.from_url(REDIS_URL)
    active = r.get(ACTIVE_SESSION_KEY)
    r.close()
    return active is not None and active.decode() != session_id

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

    cur_session_id = asset.session_id
    session_id = None
    try:
        asset.status = "processing"
        _set_stage(db, asset, "generating", 10)

        seq = _seq_num(db, asset)
        prefix = f"npc_{seq}"

        # 1. 이미지 업로드
        token = await tripo.upload_image(Path(asset.input_image_path))
        if _is_cancelled(cur_session_id):
            return

        # 2. 3D 모델 생성
        model_task_id = await tripo.create_model_task(token)
        _set_stage(db, asset, "generating", 30)
        await tripo.poll_task(model_task_id)
        if _is_cancelled(cur_session_id):
            return
        _set_stage(db, asset, "rigging", 50)

        t_rig_start = time.time()
        print(f"[char] model 완료 → rig 요청", flush=True)

        # 3. 리깅
        rig_task_id = await tripo.create_rig_task(model_task_id)
        print(f"[char] rig task 등록 완료 (+{time.time()-t_rig_start:.1f}s)", flush=True)
        await tripo.poll_task(rig_task_id)
        if _is_cancelled(cur_session_id):
            return

        t_anim_start = time.time()
        print(f"[char] rig 완료 → retarget 요청", flush=True)
        _set_stage(db, asset, "animating", 65)

        # 4. 애니메이션 — walk / idle 병렬 실행 (같은 rig_task_id 입력, 독립 출력)
        async def _run_anim(anim: dict) -> tuple[dict, Path]:
            task_id = await tripo.create_animation_task(rig_task_id, anim["preset"])
            print(f"[char] {anim['key']} task 등록 완료 (+{time.time()-t_anim_start:.1f}s)", flush=True)
            result  = await tripo.poll_task(task_id)
            glb_path = STORAGE_MODELS / f"{prefix}_{anim['key']}.glb"
            await tripo.download_glb(result, glb_path)
            return anim, glb_path

        results = await asyncio.gather(*[_run_anim(anim) for anim in ANIMATIONS])

        for anim, glb_path in results:
            db.add(AssetAnimation(
                asset_id=asset_id,
                animation_key=anim["key"],
                display_name=anim["display_name"],
                file_url=f"{BACKEND_HOST}/static/models/{glb_path.name}",
                unity_function=anim["unity_function"],
            ))
            final_glb_path = glb_path

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
