"""
charuco 파이프라인 메인 진입점.

실행 방법
---------
  conda run -n charuco python -m charuco.pipeline \\
      --images_dir data/capture_session_01 \\
      --params     data/camera_params.npz \\
      --out        output/voxels.json \\
      --method     background \\
      --upload

images_dir 구조:
  TOP.jpg   FRONT.jpg   BACK.jpg   LEFT.jpg   RIGHT.jpg
  (선택) BG_TOP.jpg ... 배경 기준 이미지

처리 순서
---------
1. 카메라 내부 파라미터 로드
2. 5방향 이미지 로드
3. ChArUco 기반 카메라 포즈 추정
4. 실루엣 마스크 추출
5. Multi-view Space Carving
6. 복셀 색상 추정
7. JSON 저장
8. (옵션) 백엔드 서버로 업로드
"""

import argparse
import os
import cv2
import numpy as np

from .calibration.camera_calibrator import load_params
from .calibration.pose_estimator    import estimate_all_poses
from .reconstruction.silhouette     import build_silhouette
from .reconstruction.voxel_grid     import VoxelGrid
from .reconstruction.voxel_carver   import run_space_carving, apply_vertical_floor_constraint
from .color.color_estimator         import estimate_colors
from .utils.json_exporter           import build_json, save_json
from .utils.visualizer              import draw_voxel_grid_top_view, draw_silhouette_overlay
from .utils.server                  import upload_voxels

VIEW_NAMES = ["TOP", "FRONT", "BACK", "LEFT", "RIGHT"]


def load_images(images_dir: str) -> tuple[dict, dict]:
    """
    지정된 디렉토리에서 뷰 이미지와 배경 이미지를 로드한다.
    BG_<VIEW>.jpg 파일이 있으면 배경 차분 방식에 사용된다.
    """
    images, backgrounds = {}, {}
    for view in VIEW_NAMES:
        for ext in ("jpg", "png"):
            path = os.path.join(images_dir, f"{view}.{ext}")
            if os.path.exists(path):
                images[view] = cv2.imread(path)
                break

        bg_path = os.path.join(images_dir, f"BG_{view}.jpg")
        if os.path.exists(bg_path):
            backgrounds[view] = cv2.imread(bg_path)

    print(f"[pipeline] 로드된 뷰: {list(images.keys())}")
    return images, backgrounds


def run_pipeline(images_dir: str,
                 params_path: str,
                 out_path: str,
                 mask_method: str = "background",
                 studs_x: int = 16,
                 studs_y: int = 16,
                 max_z:   int = 10,
                 upload:  bool = False,
                 backend_url: str = "http://localhost:8000") -> dict:
    """
    전체 파이프라인을 실행하고 복셀 JSON dict를 반환한다.

    upload=True 이면 완료 후 백엔드 서버로 자동 업로드한다.
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. 카메라 내부 파라미터
    camera_matrix, dist_coeffs = load_params(params_path)

    # 2. 이미지 로드
    images, backgrounds = load_images(images_dir)
    if not images:
        raise FileNotFoundError(f"뷰 이미지를 찾을 수 없습니다: {images_dir}")

    # 3. 포즈 추정
    poses = estimate_all_poses(images, camera_matrix, dist_coeffs)
    if len(poses) < 2:
        raise RuntimeError(f"포즈 추정 실패: {list(poses.keys())} (최소 2개 필요)")

    # 4. 실루엣 마스크 추출 + 디버그 이미지 저장
    silhouettes = {}
    for view, img in images.items():
        bg     = backgrounds.get(view)
        method = mask_method if bg is not None else "grabcut"
        sil    = build_silhouette(img, view, background=bg, method=method)
        silhouettes[view] = sil

        debug_img = draw_silhouette_overlay(img, sil.mask)
        cv2.imwrite(os.path.join(out_dir, f"mask_{view}.jpg"), debug_img)

    # 5. Space Carving
    grid = VoxelGrid(studs_x=studs_x, studs_y=studs_y, max_z=max_z)
    run_space_carving(grid, silhouettes, poses, camera_matrix, dist_coeffs)
    apply_vertical_floor_constraint(grid)

    # 6. 색상 추정
    occupied_list = grid.to_list()
    coloured_list = estimate_colors(occupied_list, grid, images, poses,
                                    camera_matrix, dist_coeffs, quantise=True)

    # 7. JSON 저장
    payload = build_json(coloured_list, studs_x, studs_y,
                          views_used=list(poses.keys()))
    save_json(payload, out_path)

    # 디버그: 탑뷰 점유 맵
    top_map = draw_voxel_grid_top_view(grid.occupied)
    cv2.imwrite(os.path.join(out_dir, "topview_debug.jpg"), top_map)

    # 8. 백엔드 업로드
    if upload:
        upload_voxels(payload, backend_url=backend_url)

    return payload


def main():
    parser = argparse.ArgumentParser(description="LEGO ChArUco 복셀 파이프라인")
    parser.add_argument("--images_dir",   required=True,   help="5방향 이미지 디렉토리")
    parser.add_argument("--params",       default="data/camera_params.npz")
    parser.add_argument("--out",          default="output/voxels.json")
    parser.add_argument("--method",       default="background",
                        choices=["background", "color", "grabcut"])
    parser.add_argument("--studs_x",      type=int, default=16)
    parser.add_argument("--studs_y",      type=int, default=16)
    parser.add_argument("--max_z",        type=int, default=10)
    parser.add_argument("--upload",       action="store_true", help="백엔드로 업로드")
    parser.add_argument("--backend_url",  default="http://localhost:8000")
    args = parser.parse_args()

    run_pipeline(
        images_dir  = args.images_dir,
        params_path = args.params,
        out_path    = args.out,
        mask_method = args.method,
        studs_x     = args.studs_x,
        studs_y     = args.studs_y,
        max_z       = args.max_z,
        upload      = args.upload,
        backend_url = args.backend_url,
    )


if __name__ == "__main__":
    main()
