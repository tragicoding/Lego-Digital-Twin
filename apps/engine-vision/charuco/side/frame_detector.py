"""
SIDE 뷰 ㄷ자형 ChArUco/ArUco 프레임 검출 및 호모그래피 계산.

물리 배치 (example2.png 참고):
  ┌──────────────────────────────┐
  │       ArUco 상단 마커들       │
  │  ┌────────────────────────┐  │
  │  │                        │  │
  │  │   레고 측면 인식 영역    │  │
  │  │                        │  │
  │  └                        ┘  │
  │  ArUco 좌측    ArUco 우측    │
  └──────────────────────────────┘
  바닥선 z=0 (설치 시 고정, 이미지에서 보이지 않음)

  - 하단 마커 없음 → z=0은 config/setup.json의 픽셀 y좌표로 고정
  - 상단 + 좌측 + 우측 마커로 호모그래피 계산
  - FRONT/BACK: x-z 평면, LEFT/RIGHT: y-z 평면

흐름:
  detect_side_u_frame_markers()
      → load_z0_from_config()
          → compute_side_homography()
              → warp_side_area()
"""

import json
import cv2
import cv2.aruco as aruco
import numpy as np

from ..calibration.charuco_board import ARUCO_DICT


def detect_side_u_frame_markers(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    ㄷ자형 프레임 (상단 + 좌측 + 우측) ArUco 마커를 검출한다.
    하단은 없으므로 z=0은 config에서 로드.

    반환: {marker_id: corners_4x2} (왜곡 보정 후 픽셀 좌표)
    """
    detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 3:
        print(f"[side] 마커 검출 부족: {len(ids) if ids is not None else 0}개 (최소 3개 필요)")
        return {}

    result = {}
    for i, mid in enumerate(ids.ravel()):
        pts = corners_list[i].reshape(4, 2)
        pts_undist = cv2.undistortPoints(
            pts.reshape(1, -1, 2).astype(np.float32),
            camera_matrix, dist_coeffs, P=camera_matrix
        ).reshape(4, 2)
        result[int(mid)] = pts_undist

    print(f"[side] 마커 검출: {sorted(result.keys())}")
    return result


def load_z0_from_config(config_path: str, view: str) -> int:
    """
    설치 시 고정한 z=0 기준선 픽셀 y좌표를 config/setup.json에서 로드한다.

    Parameters
    ----------
    config_path : config/setup.json 경로
    view        : "FRONT" | "BACK" | "LEFT" | "RIGHT"

    반환: 이미지 내 z=0에 해당하는 픽셀 y좌표 (int)
    """
    with open(config_path) as f:
        cfg = json.load(f)
    z0 = cfg["z0_pixel_y"].get(view)
    if z0 is None:
        raise KeyError(f"setup.json에 '{view}' z0 값 없음")
    print(f"[side] {view} z0 = {z0}px")
    return int(z0)


def compute_side_homography(
    marker_corners: dict[int, np.ndarray],
    top_ids: tuple[int, int],
    side_ids: tuple[int, int],
    z0_y_px: int,
    output_size: tuple[int, int],
) -> np.ndarray | None:
    """
    ㄷ자형 프레임 마커 코너 + z=0 기준선 → 사이드뷰 호모그래피 계산.

    상단 마커 2개 (top-left, top-right 역할) +
    z=0 기준선의 좌우 x좌표를 가상 코너로 사용해 4-point 호모그래피를 계산한다.

    Parameters
    ----------
    top_ids    : (좌상단 마커ID, 우상단 마커ID)
    side_ids   : (좌측 마커ID, 우측 마커ID) — 좌우 경계 x좌표 추출용
    z0_y_px    : z=0 기준선 픽셀 y좌표
    output_size: (width, height) 워핑 결과 해상도
    """
    tl_id, tr_id = top_ids
    ll_id, rl_id = side_ids

    for mid in [tl_id, tr_id, ll_id, rl_id]:
        if mid not in marker_corners:
            print(f"[side] 마커 {mid} 미검출 → 호모그래피 계산 불가")
            return None

    # 상단 마커: 각각 하단 코너(BUILD AREA 상단 경계)
    top_left_px  = marker_corners[tl_id][3]   # TL 마커의 BL 코너
    top_right_px = marker_corners[tr_id][2]   # TR 마커의 BR 코너

    # 좌우 마커에서 x 좌표 추출 (z=0 가상 코너)
    left_x  = float(np.mean(marker_corners[ll_id][:, 0]))
    right_x = float(np.mean(marker_corners[rl_id][:, 0]))

    src_pts = np.array([
        top_left_px,                      # 좌상단
        top_right_px,                     # 우상단
        [right_x, z0_y_px],              # 우하단 (z=0 기준선)
        [left_x,  z0_y_px],              # 좌하단 (z=0 기준선)
    ], dtype=np.float32)

    w, h = output_size
    dst_pts = np.array([
        [0,     0    ],
        [w - 1, 0    ],
        [w - 1, h - 1],
        [0,     h - 1],
    ], dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[side] 호모그래피 계산 실패")
        return None

    print(f"[side] 호모그래피 계산 완료")
    return H


def warp_side_area(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """H로 사이드뷰를 정렬한다. 반환: warped BGR 이미지"""
    w, h = output_size
    return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)
