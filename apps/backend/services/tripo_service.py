"""
Tripo3D API 클라이언트
image_to_model → (character only) rig → animate → download
"""
import asyncio
import httpx
from pathlib import Path
from ..core.config import TRIPO_API_KEY, TRIPO_BASE_URL

_HEADERS = {"Authorization": f"Bearer {TRIPO_API_KEY}"}


async def upload_image(image_path: Path) -> str:
    """이미지를 업로드하고 file_token을 반환한다."""
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    async with httpx.AsyncClient() as c:
        with open(image_path, "rb") as f:
            resp = await c.post(
                f"{TRIPO_BASE_URL}/upload/sts",
                headers=_HEADERS,
                files={"file": (image_path.name, f, mime)},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["data"]["image_token"]


async def create_model_task(file_token: str) -> str:
    """image_to_model 태스크를 생성하고 task_id를 반환한다."""
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json={"type": "image_to_model", "file": {"type": "png", "file_token": file_token}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]


async def create_rig_task(model_task_id: str) -> str:
    """리깅 태스크를 생성하고 task_id를 반환한다."""
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json={"type": "animate_rig", "original_model_task_id": model_task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]


async def create_animation_task(rig_task_id: str, animation: str) -> str:
    """
    애니메이션 태스크 생성.
    animation: "preset:walk" | "preset:wave" 등 Tripo 지원 preset
    """
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json={
                "type": "animate_retarget",
                "original_model_task_id": rig_task_id,
                "animation": animation,
                "out_format": "glb",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]


async def poll_task(task_id: str, interval: float = 3.0, timeout: float = 300.0) -> dict:
    """태스크 완료까지 polling하고 result dict를 반환한다."""
    elapsed = 0.0
    async with httpx.AsyncClient() as c:
        while elapsed < timeout:
            resp = await c.get(
                f"{TRIPO_BASE_URL}/task/{task_id}",
                headers=_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            status = data["status"]

            if status in ("success", "FINISHED"):
                return data
            if status in ("failed", "FAILED", "cancelled", "unknown"):
                raise RuntimeError(f"Tripo 태스크 실패: {status} (task_id={task_id})")

            await asyncio.sleep(interval)
            elapsed += interval

    raise TimeoutError(f"Tripo 태스크 타임아웃 ({timeout}s): {task_id}")


async def download_glb(result: dict, output_path: Path) -> Path:
    """GLB 파일을 다운로드한다."""
    glb_url = result["output"]["pbr_model"]
    async with httpx.AsyncClient() as c:
        resp = await c.get(glb_url, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
    return output_path
