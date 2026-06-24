"""
오브제 처리 파이프라인 (RQ Worker에서 실행)
이미지 1장 → image_to_model → GLB 저장 → DB 업데이트

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

        # 1. 태블릿 카메라에서 받은 front 이미지 1장을 Tripo에 업로드한다.
        front_path = Path(asset.input_image_path)
        front_token = await tripo.upload_image(front_path)
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        # 2. 테스트 브랜치에서는 multiview_to_model을 쓰지 않고 즉시 image_to_model로 생성한다.
        print("[obj] image_to_model: tablet camera single image", flush=True)
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
        UNITY_GENERATED_MODELS.mkdir(parents=True, exist_ok=True)
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
