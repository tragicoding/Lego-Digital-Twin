"""
charuco 파이프라인 메인 진입점.

실행 방법
---------
  conda run -n charuco python -m charuco.pipeline \\
      --images_dir data/capture_01 \\
      --params     data/camera_params.npz \\
      --config     config/setup.json \\
      --out        output/voxels.json \\
      --upload

images_dir 구조:
  TOP.jpg            (필수)
  FRONT.jpg          (선택)
  RIGHT.jpg          (선택)
  BG_TOP.jpg         (선택, 배경 기준 이미지)
  BG_FRONT.jpg       (선택)
  ...

처리 순서
---------
1. 카메라 내부 파라미터 + config 로드
2. 이미지 로드 (있는 뷰만)
3. TOP: 프레임 마커 검출 → 호모그래피 → N×N 격자 분석 (점유 + 색상)
4. SIDE: ㄷ자형 프레임 검출 → 호모그래피 → 높이맵 추출
5. TOP + SIDE 조합 → 3D 복셀 그리드
6. JSON 빌드 + 저장
7. (옵션) 백엔드 업로드
"""

import argparse
import json
import os
import cv2

from .calibration.camera_calibrator  import load_params
from .top.frame_detector             import detect_top_frame_markers, compute_top_homography, warp_build_area as warp_top
from .top.grid_analyzer              import generate_xy_grid
from .side.frame_detector            import detect_side_u_frame_markers, load_z0_from_config, compute_side_homography, warp_side_area
from .side.height_analyzer           import extract_side_silhouette, generate_height_map
from .reconstruction.voxel_combiner  import build_voxel_from_top_and_sides
from .utils.json_exporter            import build_json, save_json
from .utils.server                   import upload_voxels
from .utils.visualizer               import draw_voxel_grid_top_view

SIDE_VIEWS = ["FRONT", "BACK", "LEFT", "RIGHT"]


def load_images(images_dir: str) -> tuple[dict, dict]:
    """뷰 이미지와 배경 이미지를 로드한다. 없는 뷰는 건너뜀."""
    images, backgrounds = {}, {}
    for view in ["TOP"] + SIDE_VIEWS:
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


def run_pipeline(
    images_dir:  str,
    params_path: str,
    config_path: str,
    out_path:    str,
    upload:      bool = False,
    backend_url: str  = "http://localhost:8000",
) -> dict:

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. 내부 파라미터 + config 로드
    camera_matrix, dist_coeffs = load_params(params_path)
    with open(config_path) as f:
        cfg = json.load(f)

    grid_n      = cfg["grid_n"]
    max_z       = cfg["max_z"]
    output_size = tuple(cfg["output_size"])
    top_cfg     = cfg["top"]
    side_cfg    = cfg["side_views"]
    occ_cfg     = cfg["occupancy"]

    # 2. 이미지 로드
    images, backgrounds = load_images(images_dir)
    if "TOP" not in images:
        raise FileNotFoundError("TOP.jpg 없음 — TOP 뷰는 필수입니다.")
    views_used = list(images.keys())

    # 3. TOP 처리
    top_img = images["TOP"]
    top_bg  = backgrounds.get("TOP")

    top_markers = detect_top_frame_markers(top_img, camera_matrix, dist_coeffs)
    H_top = compute_top_homography(
        top_markers,
        corner_ids  = tuple(top_cfg["corner_ids"]),
        frame_px    = None,
        output_size = output_size,
    )
    if H_top is None:
        raise RuntimeError("TOP 호모그래피 실패 — 마커 검출 상태 확인 필요")

    warped_top    = warp_top(top_img, H_top, output_size)
    bg_warped_top = warp_top(top_bg, H_top, output_size) if top_bg is not None else None
    cv2.imwrite(os.path.join(out_dir, "warped_TOP.jpg"), warped_top)

    top_cells = generate_xy_grid(
        warped_top, grid_n,
        bg_warped            = bg_warped_top,
        occupancy_threshold  = occ_cfg["threshold"],
        saturation_threshold = occ_cfg["saturation_threshold"],
    )

    # 4. SIDE 처리
    height_maps = {}
    for view in SIDE_VIEWS:
        if view not in images:
            continue
        vcfg = side_cfg.get(view, {})
        if not vcfg:
            continue

        side_img = images[view]
        side_bg  = backgrounds.get(view)

        markers = detect_side_u_frame_markers(side_img, camera_matrix, dist_coeffs)
        z0_y    = load_z0_from_config(config_path, view)
        H_side  = compute_side_homography(
            markers,
            top_ids    = tuple(vcfg["top_ids"]),
            side_ids   = tuple(vcfg["side_ids"]),
            z0_y_px    = z0_y,
            output_size= output_size,
        )
        if H_side is None:
            print(f"[pipeline] {view} 호모그래피 실패, 스킵")
            continue

        warped_side    = warp_side_area(side_img, H_side, output_size)
        bg_warped_side = warp_side_area(side_bg,  H_side, output_size) if side_bg is not None else None
        cv2.imwrite(os.path.join(out_dir, f"warped_{view}.jpg"), warped_side)

        sil = extract_side_silhouette(warped_side, bg_warped_side,
                                       threshold=occ_cfg["threshold"])
        height_maps[view] = generate_height_map(sil, grid_n)

    # 5. 3D 복셀 조합
    voxel_grid = build_voxel_from_top_and_sides(top_cells, height_maps, grid_n, max_z)

    # 6. JSON 빌드 + 저장
    payload = build_json(top_cells, voxel_grid, height_maps, views_used, grid_n, max_z)
    save_json(payload, out_path)

    # 디버그: 탑뷰 점유맵
    top_map = draw_voxel_grid_top_view(voxel_grid.occupied)
    cv2.imwrite(os.path.join(out_dir, "topview_debug.jpg"), top_map)

    # 7. 백엔드 업로드
    if upload:
        upload_voxels(payload, backend_url=backend_url)

    return payload


def main():
    parser = argparse.ArgumentParser(description="LEGO ChArUco 격자 파이프라인")
    parser.add_argument("--images_dir",  required=True)
    parser.add_argument("--params",      default="apps/engine-vision/charuco/data/camera_params.npz")
    parser.add_argument("--config",      default="apps/engine-vision/charuco/config/setup.json")
    parser.add_argument("--out",         default="apps/engine-vision/charuco/output/voxels.json")
    parser.add_argument("--upload",      action="store_true")
    parser.add_argument("--backend_url", default="http://localhost:8000")
    args = parser.parse_args()

    run_pipeline(
        images_dir  = args.images_dir,
        params_path = args.params,
        config_path = args.config,
        out_path    = args.out,
        upload      = args.upload,
        backend_url = args.backend_url,
    )


if __name__ == "__main__":
    main()
