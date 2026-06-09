"""
오브제 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 (multiview 또는 single) → GLB 저장 → DB 업데이트

GLB 파일명: object_{순서번호}.glb
"""
import asyncio
import shutil
from pathlib import Path

from ...core.config import STORAGE_MODELS, BACKEND_HOST, UNITY_GENERATED_MODELS
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.session import Session
from ...services import tripo_service as tripo


def _seq_num(db, asset: Asset) -> int:
    count = db.query(Asset).filter(
        Asset.asset_type == asset.asset_type,
        Asset.created_at < asset.created_at,
    ).count()
    return count + 1


def _is_session_cancelled(db, session_id: str) -> bool:
    session = db.get(Session, session_id)
    return session is None or session.status == "cancelled"


def _mark_cancelled(db, asset: Asset):
    if asset.status == "completed":
        return
    asset.status = "cancelled"
    asset.stage = "cancelled"
    db.commit()


def process_object(asset_id: str):
    asyncio.run(_run(asset_id))


async def _run(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return

    session_id = None
    try:
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        asset.status = "processing"
        asset.stage = "generating"
        asset.progress = 10
        db.commit()

        seq = _seq_num(db, asset)
        prefix = f"object_{seq}"

        # 1. front 이미지 업로드
        front_path = Path(asset.input_image_path)
        front_token = await tripo.upload_image(front_path)
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        # 2. 나머지 이미지 대기 (최대 90초)
        left_path = back_path = right_path = None
        for tick in range(90):
            img_dir = front_path.parent
            if not left_path:
                c = sorted(img_dir.glob("object_left*"))
                if c: left_path = c[0]
            if not back_path:
                c = sorted(img_dir.glob("object_back*"))
                if c: back_path = c[0]
            if not right_path:
                c = sorted(img_dir.glob("object_right*"))
                if c: right_path = c[0]
            if left_path and back_path and right_path:
                print(f"[obj] 4방향 이미지 모두 수신 ({tick+1}s)", flush=True)
                break
            await asyncio.sleep(1.0)

        # 3. 3D 모델 생성 — back 이미지가 있으면 multiview, 없으면 single
        if back_path:
            found = [p.name for p in [left_path, back_path, right_path] if p]
            print(f"[obj] multiview_to_model: {found}", flush=True)
            back_token = await tripo.upload_image(back_path)
            left_token = await tripo.upload_image(left_path) if left_path else None
            right_token = await tripo.upload_image(right_path) if right_path else None
            if _is_session_cancelled(db, asset.session_id):
                _mark_cancelled(db, asset)
                return
            task_id = await tripo.create_multiview_task(
                front_token, back_token, left_token, right_token
            )
        else:
            print(f"[obj] back 이미지 없음 → image_to_model (fallback)", flush=True)
            task_id = await tripo.create_model_task(front_token, front_path)

        asset.progress = 40
        db.commit()

        result = await tripo.poll_task(task_id)
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        asset.stage = "downloading"
        asset.progress = 80
        db.commit()

        glb_path = STORAGE_MODELS / f"{prefix}.glb"
        await tripo.download_glb(result, glb_path)

        # Unity StreamingAssets로 복사
        shutil.copyfile(glb_path, UNITY_GENERATED_MODELS / glb_path.name)
        print(f"[obj] Unity 복사 완료: {glb_path.name}", flush=True)

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
