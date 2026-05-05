import shutil
from pathlib import Path

from ..core.config import UNITY_ASSETS_PATH


def deliver_to_unity(glb_path: Path) -> Path:
    """GLB 파일을 Unity External Assets 경로로 복사하고 목적지 경로를 반환한다."""
    dest = UNITY_ASSETS_PATH / glb_path.name
    shutil.copy2(glb_path, dest)
    return dest
