"""
오브제 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 (multiview 또는 single) → GLB 저장 → DB 업데이트

GLB 파일명: object_{session_id}_{asset_id}.glb
"""
import asyncio
import shutil
import re
from pathlib import Path

from ...core.config import STORAGE_MODELS, BACKEND_HOST, UNITY_GENERATED_MODELS
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.session import Session
from ...services import tripo_service as tripo


SIDE_VIEWS = ("left", "back", "right")


def _safe_name_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return safe[:80] or "unknown"


def _model_prefix(asset: Asset) -> str:
    return f"{asset.asset_type}_{_safe_name_part(asset.session_id)}_{_safe_name_part(asset.id)}"


def _capture_prefix_from_front(front_path: Path, asset_type: str) -> str | None:
    """새 파일명(object_<capture_id>_front.jpg)이면 object_<capture_id>를 돌려준다."""
    stem = front_path.stem
    suffix = "_front"
    if not (stem.startswith(f"{asset_type}_") and stem.endswith(suffix)):
        return None

    prefix = stem[: -len(suffix)]
    # 예전 Windows relay 파일명은 object_object_front.jpg처럼 저장됐다.
    # 이 경우 side 파일이 object_left_object_left.jpg 형태라 capture prefix로 쓰면 안 된다.
    if prefix == f"{asset_type}_{asset_type}":
        return None
    return prefix


def _newest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _find_view_path(img_dir: Path, asset_type: str, view: str, capture_prefix: str | None) -> Path | None:
    if capture_prefix:
        return _newest(list(img_dir.glob(f"{capture_prefix}_{view}.*")))
    # 기존 파일명 호환: object_left_object_left.jpg
    return _newest(list(img_dir.glob(f"{asset_type}_{view}_*")))


def _is_session_cancelled(db, session_id: str) -> bool:
    session = db.get(Session, session_id)
    return session is None or session.status == "cancelled"


def _should_stop(db, asset: Asset) -> bool:
    db.refresh(asset)
    return asset.status in ("cancelled", "superseded") or _is_session_cancelled(db, asset.session_id)


def _mark_cancelled(db, asset: Asset):
    if asset.status in ("completed", "superseded"):
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
        if _should_stop(db, asset):
            _mark_cancelled(db, asset)
            return

        asset.status = "processing"
        asset.stage = "generating"
        asset.progress = 10
        db.commit()

        prefix = _model_prefix(asset)

        # 1. front 이미지 업로드
        front_path = Path(asset.input_image_path)
        front_token = await tripo.upload_image(front_path)
        if _should_stop(db, asset):
            _mark_cancelled(db, asset)
            return

        # 2. 나머지 이미지 대기 (최대 90초)
        capture_prefix = _capture_prefix_from_front(front_path, asset.asset_type)
        left_path = back_path = right_path = None
        for tick in range(90):
            if _should_stop(db, asset):
                _mark_cancelled(db, asset)
                return
            img_dir = front_path.parent
            if not left_path:
                left_path = _find_view_path(img_dir, asset.asset_type, "left", capture_prefix)
            if not back_path:
                back_path = _find_view_path(img_dir, asset.asset_type, "back", capture_prefix)
            if not right_path:
                right_path = _find_view_path(img_dir, asset.asset_type, "right", capture_prefix)
            if left_path and back_path and right_path:
                print(f"[obj] {asset.session_id}/{asset.id} 4방향 이미지 모두 수신 ({tick+1}s)", flush=True)
                break
            await asyncio.sleep(1.0)

        # 3. 3D 모델 생성
        # capture_id가 있는 카메라 촬영은 반드시 같은 촬영 묶음의 4방향 이미지가 있어야 한다.
        if capture_prefix and not (left_path and back_path and right_path):
            missing = [view for view, path in (("left", left_path), ("back", back_path), ("right", right_path)) if not path]
            raise RuntimeError(
                f"missing side view(s) for capture {capture_prefix}: {', '.join(missing)}"
            )
        if not capture_prefix and any((left_path, back_path, right_path)) and not (left_path and back_path and right_path):
            missing = [view for view, path in (("left", left_path), ("back", back_path), ("right", right_path)) if not path]
            raise RuntimeError(f"incomplete side views for {asset.session_id}/{asset.id}: {', '.join(missing)}")

        # 기존 수동/레거시 업로드는 back 이미지가 있으면 multiview, 없으면 single fallback
        if back_path:
            found = [p.name for p in [left_path, back_path, right_path] if p]
            print(f"[obj] {asset.session_id}/{asset.id} multiview_to_model: {found}", flush=True)
            back_token = await tripo.upload_image(back_path)
            left_token = await tripo.upload_image(left_path) if left_path else None
            right_token = await tripo.upload_image(right_path) if right_path else None
            if _should_stop(db, asset):
                _mark_cancelled(db, asset)
                return
            task_id = await tripo.create_multiview_task(
                front_token, back_token, left_token, right_token
            )
        else:
            print(f"[obj] {asset.session_id}/{asset.id} back 이미지 없음 → image_to_model (fallback)", flush=True)
            task_id = await tripo.create_model_task(front_token, front_path)

        asset.progress = 40
        db.commit()

        result = await tripo.poll_task(task_id)
        if _should_stop(db, asset):
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
        db.refresh(asset)
        if asset.status != "superseded":
            asset.status = "failed"
            asset.error_message = str(e) or repr(e)
            db.commit()
    finally:
        db.close()

    if session_id:
        from ...services.event_service import check_and_notify
        check_and_notify(session_id)
