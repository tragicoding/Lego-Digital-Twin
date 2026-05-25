"""
캐릭터 처리 파이프라인 (RQ Worker에서 실행)
이미지 → 3D 생성 → 리깅(FBX) → FBX 저장 → DB 업데이트

파이프라인 변경 이유:
- Tripo 리깅 결과를 FBX로 받아 Unity Animator 자동 리타겟팅에 사용
- 모든 캐릭터의 리깅 구조가 동일하므로 리타겟팅이 가능
- 애니메이션은 Unity에서 처리 (Backend에서 animate_retarget 미수행)

저장 파일:
  npc_{순서번호}_rigged.fbx  — 리깅된 캐릭터 (Unity 리타겟팅용)

[텍스쳐 전략 — 테스트 후 결정]
  Case A: FBX에 텍스쳐 임베딩 O  → Unity 로더 설정만 수정, 백엔드 추가 없음
  Case B: FBX에 텍스쳐 없음      → image_to_model output의 PNG URL만 추가 다운로드
  (GLB 전체 재다운로드 X — 메시 중복 다운로드 방지)

  image_to_model output 에서 추출 가능한 텍스쳐 키 (API 확인 필요):
    output.base_color / output.normal / output.metallic_roughness 등
"""
import asyncio
import time
from pathlib import Path

from redis import Redis

from ...core.config import STORAGE_MODELS, BACKEND_HOST, REDIS_URL
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...services import tripo_service as tripo

ACTIVE_SESSION_KEY = "lego:active_session"


def _is_cancelled(session_id: str) -> bool:
    """현재 세션이 active session이 아니면 True (새 세션이 시작된 것)."""
    r = Redis.from_url(REDIS_URL)
    active = r.get(ACTIVE_SESSION_KEY)
    r.close()
    return active is not None and active.decode() != session_id


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

        # 2. 3D 모델 생성 (image_to_model)
        model_task_id = await tripo.create_model_task(token)
        _set_stage(db, asset, "generating", 30)
        model_result = await tripo.poll_task(model_task_id)
        # TODO: 텍스쳐 누락 시 여기서 model_result["output"]의 키를 확인해
        #       PNG 텍스쳐 URL(base_color 등)을 별도 다운로드한다.
        print(f"[char] model output keys: {list(model_result.get('output', {}).keys())}", flush=True)
        if _is_cancelled(cur_session_id):
            return

        _set_stage(db, asset, "rigging", 50)
        t_rig_start = time.time()
        print(f"[char] model 완료 → rig(FBX) 요청", flush=True)

        # 3. 리깅 — out_format="fbx" 로 요청 (Unity Animator 리타겟팅용)
        rig_task_id = await tripo.create_rig_task(model_task_id, out_format="fbx")
        print(f"[char] rig task 등록 완료 (+{time.time()-t_rig_start:.1f}s)", flush=True)
        rig_result = await tripo.poll_task(rig_task_id)
        print(f"[char] rig output keys: {list(rig_result.get('output', {}).keys())}", flush=True)
        if _is_cancelled(cur_session_id):
            return

        _set_stage(db, asset, "downloading", 80)

        # 4. 리깅된 FBX 다운로드
        fbx_path = STORAGE_MODELS / f"{prefix}_rigged.fbx"
        await tripo.download_fbx(rig_result, fbx_path)
        print(f"[char] FBX 저장 완료: {fbx_path.name} (+{time.time()-t_rig_start:.1f}s)", flush=True)

        # 5. DB 업데이트
        asset.model_url = f"{BACKEND_HOST}/static/models/{fbx_path.name}"
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
