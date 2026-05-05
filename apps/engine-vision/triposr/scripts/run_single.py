"""
단일 이미지 → TripoSR 3D 재구성 (수동 실행용)
사용법: python scripts/run_single.py <image_path> [--mc-resolution 256]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TRIPOSR_ROOT = Path(__file__).parent.parent / "TripoSR"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def main():
    parser = argparse.ArgumentParser(description="TripoSR 단일 이미지 3D 재구성")
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--mc-resolution", type=int, default=256, help="Marching Cubes 해상도 (기본: 256)")
    parser.add_argument("--chunk-size", type=int, default=8192, help="청크 크기 (기본: 8192)")
    parser.add_argument("--remove-bg", action="store_true", help="배경 제거 활성화")
    parser.add_argument("--output-dir", default=None, help="출력 디렉토리 (기본: output/<이미지명>)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"오류: 이미지 파일 없음 — {image_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRIPOSR_ROOT / "run.py"),
        str(image_path),
        "--output-dir", str(output_dir),
        "--device", "cuda",
        "--chunk-size", str(args.chunk_size),
        "--mc-resolution", str(args.mc_resolution),
    ]
    if not args.remove_bg:
        cmd.append("--no-remove-bg")

    print(f"입력: {image_path}")
    print(f"출력: {output_dir}")
    print(f"실행: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(TRIPOSR_ROOT))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n완료 ({elapsed:.1f}초)")
        print(f"결과 파일: {list(output_dir.glob('*'))}")
    else:
        print(f"\n실패 (returncode={result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    main()
