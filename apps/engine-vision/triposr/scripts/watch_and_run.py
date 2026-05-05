"""
Windows 카메라 → WSL2 자동 TripoSR 파이프라인
Windows Camera Roll 폴더를 감시하다가 새 이미지 감지 시 자동으로 3D 재구성 실행
"""

import os
import sys
import time
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

# === 경로 설정 ===
WINDOWS_CAMERA_ROLL = Path("/mnt/c/Users/wlstj/OneDrive/사진/Camera Roll")
WINDOWS_SAVED_PICTURES = Path("/mnt/c/Users/wlstj/OneDrive/사진/Saved Pictures")

TRIPOSR_ROOT = Path(__file__).parent.parent / "TripoSR"
INPUT_DIR = Path(__file__).parent.parent / "input"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LOG_DIR = Path(__file__).parent.parent / "logs"

WATCH_DIRS = [WINDOWS_CAMERA_ROLL, WINDOWS_SAVED_PICTURES]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 처리 중 중복 방지용 set
processing_files = set()

# === 로깅 설정 ===
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_triposr(image_path: Path) -> Path | None:
    """TripoSR로 단일 이미지 → 3D mesh 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = OUTPUT_DIR / f"{image_path.stem}_{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 입력 이미지를 input/ 에 복사
    dest_input = INPUT_DIR / image_path.name
    shutil.copy2(image_path, dest_input)
    logger.info(f"입력 이미지 복사: {dest_input}")

    cmd = [
        sys.executable,
        str(TRIPOSR_ROOT / "run.py"),
        str(dest_input),
        "--output-dir", str(output_subdir),
        "--device", "cuda",
        "--chunk-size", "8192",  # 8GB VRAM 최적화
        "--mc-resolution", "256",  # RTX4070 8GB 적합 해상도
        "--no-remove-bg",  # 이미 배경 없는 경우 스킵 (옵션)
    ]

    logger.info(f"TripoSR 실행: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(TRIPOSR_ROOT),
            timeout=300,  # 5분 타임아웃
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info(f"완료 ({elapsed:.1f}초): {output_subdir}")
            return output_subdir
        else:
            logger.error(f"TripoSR 실패 (returncode={result.returncode}):\n{result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("TripoSR 타임아웃 (5분 초과)")
        return None
    except Exception as e:
        logger.error(f"예외 발생: {e}")
        return None


class CameraRollHandler(FileSystemEventHandler):
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            path = Path(event.src_path)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                if str(path) not in processing_files:
                    processing_files.add(str(path))
                    logger.info(f"새 이미지 감지: {path.name}")
                    # 파일 쓰기 완료 대기 (1초)
                    time.sleep(1.0)
                    result = run_triposr(path)
                    if result:
                        logger.info(f"3D 메시 저장 위치: {result}")
                    processing_files.discard(str(path))

    def on_moved(self, event):
        # OneDrive sync 완료 시 moved 이벤트 발생하는 경우 처리
        if not event.is_directory:
            path = Path(event.dest_path)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                if str(path) not in processing_files:
                    processing_files.add(str(path))
                    logger.info(f"새 이미지 동기화 완료: {path.name}")
                    time.sleep(1.0)
                    result = run_triposr(path)
                    if result:
                        logger.info(f"3D 메시 저장 위치: {result}")
                    processing_files.discard(str(path))


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # PyTorch/CUDA 확인
    try:
        import torch
        logger.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch 미설치 — TripoSR 실행 불가")

    observer = Observer()
    handler = CameraRollHandler()

    active_dirs = []
    for watch_dir in WATCH_DIRS:
        if watch_dir.exists():
            observer.schedule(handler, str(watch_dir), recursive=False)
            active_dirs.append(watch_dir)
            logger.info(f"감시 시작: {watch_dir}")
        else:
            logger.warning(f"경로 없음 (건너뜀): {watch_dir}")

    if not active_dirs:
        logger.error("감시할 폴더가 없습니다. Windows 경로를 확인하세요.")
        sys.exit(1)

    observer.start()
    logger.info("파이프라인 대기 중... (Ctrl+C로 종료)")
    logger.info(f"로그: {log_file}")

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    logger.info("파이프라인 종료")


if __name__ == "__main__":
    main()
