from fastapi import APIRouter, HTTPException
import httpx

from ...core.config import WINDOWS_CAMERA_URL

router = APIRouter(prefix="/camera", tags=["camera"])


@router.get("/health")
def camera_health():
    """Windows USB camera service readiness proxy."""
    url = f"{WINDOWS_CAMERA_URL.rstrip('/')}/health"
    try:
        response = httpx.get(url, timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Windows camera service 연결 실패: {exc}",
        ) from exc
