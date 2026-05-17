"""
Mock 처리 파이프라인 — Tripo API 없이 단계별 시뮬레이션.
test_app 브랜치 전용. 실제 파일 없이 전체 흐름을 테스트한다.
"""
import asyncio
from pathlib import Path

from ...core.config import STORAGE_MODELS, BACKEND_HOST
from ...core.database import SessionLocal
from ...models.asset import Asset
from ...models.asset_animation import AssetAnimation

# 이미 생성된 GLB가 있으면 재사용, 없으면 placeholder URL
_FALLBACK_GLB = next(STORAGE_MODELS.glob("*.glb"), None)
_MODEL_URL_BASE = f"{BACKEND_HOST}/static/models"


def _fake_model_url(asset_id: str, suffix: str) -> str:
    if _FALLBACK_GLB:
        return f"{_MODEL_URL_BASE}/{_FALLBACK_GLB.name}"
    return f"{_MODEL_URL_BASE}/{asset_id}_{suffix}.glb"


def _set_stage(db, asset: Asset, stage: str, progress: int):
    asset.stage = stage
    asset.progress = progress
    db.commit()


# ── 공통 헬퍼 ──────────────────────────────────────────────

async def _simulate(db, asset: Asset, steps: list[tuple[str, int, float]]):
    """steps = [(stage, progress, sleep_sec), ...]"""
    asset.status = "processing"
    db.commit()
    for stage, progress, delay in steps:
        _set_stage(db, asset, stage, progress)
        await asyncio.sleep(delay)


# ── 캐릭터 ──────────────────────────────────────────────────

def process_mock_character(asset_id: str):
    asyncio.run(_mock_character(asset_id))


async def _mock_character(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return
    try:
        await _simulate(db, asset, [
            ("generating", 15, 2),
            ("generating", 35, 2),
            ("rigging",    55, 2),
            ("animating",  70, 2),
            ("animating",  85, 1),
            ("downloading", 95, 1),
        ])

        # 애니메이션 메타데이터 저장
        for anim in [
            {"key": "walk",  "display_name": "걷기",    "unity_function": "animation_walk"},
            {"key": "hello", "display_name": "인사_01", "unity_function": "animation_Hello"},
        ]:
            db.add(AssetAnimation(
                asset_id=asset_id,
                animation_key=anim["key"],
                display_name=anim["display_name"],
                unity_function=anim["unity_function"],
                file_url=_fake_model_url(asset_id, anim["key"]),
            ))

        asset.model_url = _fake_model_url(asset_id, "character_animated")
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


# ── 건물 ────────────────────────────────────────────────────

def process_mock_building(asset_id: str):
    asyncio.run(_mock_building(asset_id))


async def _mock_building(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return
    try:
        await _simulate(db, asset, [
            ("generating", 20, 2),
            ("generating", 60, 2),
            ("downloading", 90, 1),
        ])
        asset.model_url = _fake_model_url(asset_id, "building")
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


# ── 자동차 ──────────────────────────────────────────────────

def process_mock_vehicle(asset_id: str):
    asyncio.run(_mock_vehicle(asset_id))


async def _mock_vehicle(asset_id: str):
    db = SessionLocal()
    asset = db.get(Asset, asset_id)
    if not asset:
        return
    try:
        await _simulate(db, asset, [
            ("generating", 20, 2),
            ("generating", 60, 2),
            ("downloading", 90, 1),
        ])
        asset.model_url = _fake_model_url(asset_id, "vehicle")
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
