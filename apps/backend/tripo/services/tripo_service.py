"""
Tripo3D API 클라이언트
https://api.tripo3d.ai/v2/openapi
"""

import asyncio
import httpx
from pathlib import Path

from ..core.config import TRIPO_BASE_URL


class TripoService:
    def __init__(self, api_key: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def upload_image(self, image_bytes: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower().lstrip(".")
        mime = "image/png" if ext == "png" else "image/jpeg"
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{TRIPO_BASE_URL}/upload/sts",
                headers=self._headers,
                files={"file": (filename, image_bytes, mime)},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["data"]["image_token"]

    async def create_task(self, image_token: str) -> str:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{TRIPO_BASE_URL}/task",
                headers={**self._headers, "Content-Type": "application/json"},
                json={
                    "type": "image_to_model",
                    "file": {
                        "type": "png",
                        "file_token": image_token,
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["data"]["task_id"]

    async def poll_task(
        self,
        task_id: str,
        interval: float = 3.0,
        timeout: float = 300.0,
    ) -> dict:
        elapsed = 0.0
        async with httpx.AsyncClient() as c:
            while elapsed < timeout:
                resp = await c.get(
                    f"{TRIPO_BASE_URL}/task/{task_id}",
                    headers=self._headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                status = data["status"]

                if status in ("success", "FINISHED"):
                    return data
                if status in ("failed", "FAILED", "cancelled", "unknown"):
                    raise RuntimeError(f"Tripo 태스크 실패: {status}")

                await asyncio.sleep(interval)
                elapsed += interval

        raise TimeoutError(f"Tripo 태스크 타임아웃 ({timeout}s)")

    async def download_model(self, result: dict, output_path: Path) -> Path:
        glb_url = result["output"]["pbr_model"]
        async with httpx.AsyncClient() as c:
            resp = await c.get(glb_url, timeout=120, follow_redirects=True)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
        return output_path

    async def run(
        self, image_bytes: bytes, filename: str, output_path: Path
    ) -> Path:
        token = await self.upload_image(image_bytes, filename)
        tripo_task_id = await self.create_task(token)
        result = await self.poll_task(tripo_task_id)
        return await self.download_model(result, output_path)
