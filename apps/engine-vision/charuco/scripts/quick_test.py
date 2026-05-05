"""
빠른 테스트 스크립트.

카메라 캘리브레이션 없이도 실행 가능.
근사 카메라 파라미터를 사용하므로 정밀도는 낮지만
마커 검출 여부, 호모그래피, 격자 분석, 3D 미리보기를 한 번에 확인할 수 있다.

실행 (charuco 디렉토리에서):
  conda run -n charuco python scripts/quick_test.py \\
      --capture_dir data/capture_test \\
      --out         output/quick_test

결과:
  output/quick_test/markers_TOP.jpg    마커 검출 시각화
  output/quick_test/markers_RIGHT.jpg
  output/quick_test/warped_TOP.jpg     호모그래피 워핑 결과
  output/quick_test/topview_debug.jpg  격자 점유맵
  output/quick_test/voxels.json        복셀 데이터
  → Open3D 3D 미리보기 창 팝업
"""

import argparse
import os
import sys
import json

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from charuco.calibration.charuco_board import ARUCO_DICT
from charuco.top.frame_detector        import detect_top_frame_markers, compute_top_homography, warp_build_area
from charuco.top.grid_analyzer         import generate_xy_grid
from charuco.side.frame_detector       import detect_side_u_frame_markers, compute_side_homography, warp_side_area
from charuco.side.height_analyzer      import extract_side_silhouette, generate_height_map
from charuco.reconstruction.voxel_combiner import build_voxel_from_top_and_sides
from charuco.utils.json_exporter       import build_json, save_json
from charuco.utils.visualizer          import draw_voxel_grid_top_view


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/setup.json")


def make_approx_camera_params(image: np.ndarray):
    """캘리브레이션 없이 근사 카메라 파라미터를 생성한다."""
    h, w = image.shape[:2]
    f  = max(h, w)          # 초점거리 근사 (픽셀)
    K  = np.array([[f, 0, w/2],
                   [0, f, h/2],
                   [0, 0,   1]], dtype=np.float64)
    D  = np.zeros(5, dtype=np.float64)
    return K, D


def draw_markers(image: np.ndarray, marker_corners: dict) -> np.ndarray:
    """검출된 마커를 이미지에 오버레이한다."""
    vis = image.copy()
    for mid, corners in marker_corners.items():
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        cx = int(corners[:, 0].mean())
        cy = int(corners[:, 1].mean())
        cv2.putText(vis, str(mid), (cx - 15, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(vis, f"Detected: {len(marker_corners)} markers",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)
    return vis


def load_image(base_dir: str, view: str):
    for ext in ("jpg", "png", "jpeg"):
        p = os.path.join(base_dir, f"{view}.{ext}")
        if os.path.exists(p):
            return cv2.imread(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_dir", default="data/capture_test")
    parser.add_argument("--out",         default="output/quick_test")
    parser.add_argument("--grid_n",      type=int, default=32)
    parser.add_argument("--max_z",       type=int, default=12)
    parser.add_argument("--no_3d",       action="store_true", help="3D 미리보기 건너뜀")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    output_size = tuple(cfg["output_size"])

    # ── TOP 처리 ─────────────────────────────────────────────
    top_img = load_image(args.capture_dir, "TOP")
    if top_img is None:
        print("[ERROR] TOP.jpg 없음"); sys.exit(1)

    K, D = make_approx_camera_params(top_img)

    top_markers = detect_top_frame_markers(top_img, K, D)
    vis_top = draw_markers(top_img, top_markers)
    cv2.imwrite(os.path.join(args.out, "markers_TOP.jpg"), vis_top)
    print(f"[TOP] 마커 {len(top_markers)}개 검출: {sorted(top_markers.keys())}")

    if len(top_markers) < 4:
        print("[TOP] 마커 부족 — 호모그래피 불가. 조명/각도 조정 필요")
        print(f"  → markers_TOP.jpg 확인: {args.out}/markers_TOP.jpg")
        sys.exit(1)

    H_top = compute_top_homography(
        top_markers,
        corner_ids  = tuple(cfg["top"]["corner_ids"]),
        frame_px    = None,
        output_size = output_size,
    )
    if H_top is None:
        print("[TOP] 호모그래피 실패 — corner_ids 확인 필요")
        print(f"  현재 corner_ids: {cfg['top']['corner_ids']}")
        print(f"  검출된 마커 ID: {sorted(top_markers.keys())}")
        sys.exit(1)

    warped_top = warp_build_area(top_img, H_top, output_size)
    cv2.imwrite(os.path.join(args.out, "warped_TOP.jpg"), warped_top)
    print(f"[TOP] 호모그래피 완료 → warped_TOP.jpg")

    top_cells = generate_xy_grid(warped_top, args.grid_n)
    occupied_count = sum(1 for c in top_cells if c.occupied)
    print(f"[TOP] 격자 분석: {args.grid_n}×{args.grid_n}, 점유 셀 {occupied_count}개")

    # ── SIDE 처리 (RIGHT 또는 FRONT) ─────────────────────────
    height_maps = {}
    views_used  = ["TOP"]

    for view in ["FRONT", "BACK", "LEFT", "RIGHT"]:
        side_img = load_image(args.capture_dir, view)
        if side_img is None:
            continue

        vcfg = cfg["side_views"].get(view, {})
        if not vcfg:
            continue

        K_s, D_s = make_approx_camera_params(side_img)
        markers = detect_side_u_frame_markers(side_img, K_s, D_s)
        vis_side = draw_markers(side_img, markers)
        cv2.imwrite(os.path.join(args.out, f"markers_{view}.jpg"), vis_side)
        print(f"[{view}] 마커 {len(markers)}개 검출: {sorted(markers.keys())}")

        z0_y = cfg["z0_pixel_y"].get(view, side_img.shape[0] - 20)
        H_side = compute_side_homography(
            markers,
            top_ids    = tuple(vcfg["top_ids"]),
            side_ids   = tuple(vcfg["side_ids"]),
            z0_y_px    = z0_y,
            output_size= output_size,
        )
        if H_side is None:
            print(f"[{view}] 호모그래피 실패, 스킵")
            continue

        warped_side = warp_side_area(side_img, H_side, output_size)
        cv2.imwrite(os.path.join(args.out, f"warped_{view}.jpg"), warped_side)

        sil = extract_side_silhouette(warped_side)
        height_maps[view] = generate_height_map(sil, args.grid_n)
        views_used.append(view)
        print(f"[{view}] 높이맵 추출 완료")

    # ── 3D 복셀 조합 ──────────────────────────────────────────
    voxel_grid = build_voxel_from_top_and_sides(
        top_cells, height_maps, args.grid_n, args.max_z
    )

    # 탑뷰 점유맵 저장
    top_map = draw_voxel_grid_top_view(voxel_grid.occupied)
    cv2.imwrite(os.path.join(args.out, "topview_debug.jpg"), top_map)

    # JSON 저장
    payload = build_json(top_cells, voxel_grid, height_maps, views_used,
                         args.grid_n, args.max_z)
    json_path = os.path.join(args.out, "voxels.json")
    save_json(payload, json_path)

    print(f"\n[결과]")
    print(f"  마커 시각화: {args.out}/markers_*.jpg")
    print(f"  워핑 결과:   {args.out}/warped_*.jpg")
    print(f"  탑뷰 점유맵: {args.out}/topview_debug.jpg")
    print(f"  복셀 JSON:   {json_path} ({voxel_grid.occupied_count}개)")

    # ── Open3D 3D 미리보기 ────────────────────────────────────
    if not args.no_3d:
        preview_3d(voxel_grid, top_cells)


def preview_3d(voxel_grid, top_cells):
    """Open3D로 복셀을 3D 점군으로 시각화한다."""
    try:
        import open3d as o3d
    except ImportError:
        print("[3D] open3d 없음 — conda run -n charuco pip install open3d")
        return

    from charuco.reconstruction.voxel_grid import STUD_PITCH, BRICK_HEIGHT

    # 색상 매핑
    color_lookup = {(c.x, c.y): (c.r / 255, c.g / 255, c.b / 255)
                    for c in top_cells if c.occupied}

    pts, colors = [], []
    for xi in range(voxel_grid.studs_x):
        for yi in range(voxel_grid.studs_y):
            for zi in range(voxel_grid.max_z):
                if voxel_grid.occupied[xi, yi, zi]:
                    x = xi * STUD_PITCH
                    y = yi * STUD_PITCH
                    z = zi * BRICK_HEIGHT
                    pts.append([x, y, z])
                    r, g, b = color_lookup.get((xi, yi), (0.6, 0.6, 0.6))
                    colors.append([r, g, b])

    if not pts:
        print("[3D] 점유 복셀 없음")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # 바닥 그리드 (시각적 기준)
    grid_lines = _make_grid(voxel_grid.studs_x, voxel_grid.studs_y)

    print("[3D] Open3D 미리보기 — 창 닫으면 종료")
    o3d.visualization.draw_geometries(
        [pcd, grid_lines],
        window_name = "LEGO Voxel Preview",
        width=900, height=700,
    )


def _make_grid(nx, ny):
    import open3d as o3d
    from charuco.reconstruction.voxel_grid import STUD_PITCH
    lines, pts = [], []
    for xi in range(nx + 1):
        pts += [[xi * STUD_PITCH, 0, 0], [xi * STUD_PITCH, ny * STUD_PITCH, 0]]
        lines.append([len(pts)-2, len(pts)-1])
    for yi in range(ny + 1):
        pts += [[0, yi * STUD_PITCH, 0], [nx * STUD_PITCH, yi * STUD_PITCH, 0]]
        lines.append([len(pts)-2, len(pts)-1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0.7,0.7,0.7]] * len(lines))
    return ls


if __name__ == "__main__":
    main()
