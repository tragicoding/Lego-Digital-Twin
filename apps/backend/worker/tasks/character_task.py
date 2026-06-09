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

import shutil

from ...core.config import STORAGE_MODELS, BACKEND_HOST, UNITY_GENERATED_MODELS
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.session import Session
from ...services import tripo_service as tripo


def _set_stage(db, asset: Asset, stage: str, progress: int):
    asset.stage = stage
    asset.progress = progress
    db.commit()


def _is_session_cancelled(db, session_id: str) -> bool:
    session = db.get(Session, session_id)
    return session is None or session.status == "cancelled"


def _mark_cancelled(db, asset: Asset):
    if asset.status == "completed":
        return
    asset.status = "cancelled"
    asset.stage = "cancelled"
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
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        asset.status = "processing"
        _set_stage(db, asset, "generating", 10)

        seq = _seq_num(db, asset)
        prefix = f"npc_{seq}"

        # 1. 이미지 업로드
        front_path = Path(asset.input_image_path)
        front_token = await tripo.upload_image(front_path)
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        # 2. 나머지 이미지 대기 (최대 90초) — left/back/right 업로드 시간 확보
        left_path = back_path = right_path = None
        for tick in range(90):
            img_dir = front_path.parent
            if not left_path:
                c = sorted(img_dir.glob("character_left*"))
                if c:
                    left_path = c[0]
            if not back_path:
                c = sorted(img_dir.glob("character_back*"))
                if c:
                    back_path = c[0]
            if not right_path:
                c = sorted(img_dir.glob("character_right*"))
                if c:
                    right_path = c[0]
            if left_path and back_path and right_path:
                print(f"[char] 4방향 이미지 모두 수신 ({tick+1}s)", flush=True)
                break
            await asyncio.sleep(1.0)

        # 3. 3D 모델 생성 — back 이미지가 있으면 multiview, 없으면 single
        if back_path:
            found = [p.name for p in [left_path, back_path, right_path] if p]
            print(f"[char] multiview_to_model: {found}", flush=True)
            back_token = await tripo.upload_image(back_path)
            left_token = await tripo.upload_image(left_path) if left_path else None
            right_token = await tripo.upload_image(right_path) if right_path else None
            if _is_session_cancelled(db, asset.session_id):
                _mark_cancelled(db, asset)
                return
            model_task_id = await tripo.create_multiview_task(
                front_token, back_token, left_token, right_token
            )
        else:
            print(f"[char] back 이미지 없음 → image_to_model (fallback)", flush=True)
            model_task_id = await tripo.create_model_task(front_token, front_path)

        _set_stage(db, asset, "generating", 30)
        model_result = await tripo.poll_task(model_task_id)
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        # 2-1. pbr_model GLB 다운로드 — 텍스쳐 소스
        #      animate_rig FBX 에는 텍스쳐가 포함되지 않음 (확인됨).
        #      image_to_model 은 이미 완료된 태스크이므로 추가 API 비용 없음.
        #      Unity 에서 glTFast 로 이 GLB 를 로드해 Material/Texture 를 추출,
        #      리타겟팅된 FBX 캐릭터에 적용한다.
        texture_glb_path = STORAGE_MODELS / f"{prefix}_texture.glb"
        await tripo.download_glb(model_result, texture_glb_path)
        print(f"[char] 텍스쳐 GLB 저장: {texture_glb_path.name}", flush=True)

        _set_stage(db, asset, "rigging", 50)
        t_rig_start = time.time()
        print(f"[char] model 완료 → rig(FBX) 요청", flush=True)

        # 3. 리깅 — out_format="fbx" (Unity Animator 리타겟팅용)
        rig_task_id = await tripo.create_rig_task(model_task_id, out_format="fbx")
        print(f"[char] rig task 등록 완료 (+{time.time()-t_rig_start:.1f}s)", flush=True)
        rig_result = await tripo.poll_task(rig_task_id)
        if _is_session_cancelled(db, asset.session_id):
            _mark_cancelled(db, asset)
            return

        _set_stage(db, asset, "downloading", 80)

        # 4. 리깅된 FBX 다운로드
        fbx_path = STORAGE_MODELS / f"{prefix}_rigged.fbx"
        await tripo.download_fbx(rig_result, fbx_path)
        print(f"[char] FBX 저장 완료: {fbx_path.name} (+{time.time()-t_rig_start:.1f}s)", flush=True)

        # 4-1. Unity StreamingAssets로 복사
        shutil.copy(fbx_path, UNITY_GENERATED_MODELS / fbx_path.name)
        shutil.copy(texture_glb_path, UNITY_GENERATED_MODELS / texture_glb_path.name)
        print(f"[char] Unity 복사 완료: {UNITY_GENERATED_MODELS}", flush=True)

        # 5. DB 업데이트
        #    model_url     → FBX (리타겟팅용 리깅 캐릭터)
        #    thumbnail_url → texture GLB (pbr_model, Unity 텍스쳐 소스)
        asset.model_url = f"{BACKEND_HOST}/static/models/{fbx_path.name}"
        asset.thumbnail_url = f"{BACKEND_HOST}/static/models/{texture_glb_path.name}"
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
