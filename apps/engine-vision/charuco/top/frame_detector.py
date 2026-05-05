"""
TOP 뷰 사각형 ChArUco/ArUco 프레임 검출 및 호모그래피 계산.

물리 배치 (example.png 참고):
  ┌────────────────────────────────┐
  │  ArUco 마커들 (외곽 프레임)     │
  │  ┌──────────────────────────┐  │
  │  │    LEGO BUILD AREA        │  │
  │  └──────────────────────────┘  │
  └────────────────────────────────┘

  - 프레임 마커 최소 4개 (각 모서리 인근)
  - 마커 ID로 위치 구분: 설치 시 setup.json에 코너 ID 정의
  - 호모그래피: 이미지 픽셀 → 정규화된 LEGO BUILD AREA 좌표

흐름:
  detect_top_frame_markers()
      → compute_top_homography()
          → warp_build_area()
"""

import cv2
import cv2.aruco as aruco
import numpy as np

from ..calibration.charuco_board import ARUCO_DICT


def detect_top_frame_markers(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    TOP 뷰 이미지에서 ArUco 마커를 검출한다.

    반환: {marker_id: corners_4x2}  (픽셀 좌표, 왜곡 보정 후)
    마커가 4개 미만이면 빈 dict 반환.
    """
    detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        print(f"[top] 마커 검출 부족: {len(ids) if ids is not None else 0}개 (최소 4개 필요)")
        return {}

    # 렌즈 왜곡 보정
    result = {}
    for i, mid in enumerate(ids.ravel()):
        pts = corners_list[i].reshape(4, 2)
        pts_undist = cv2.undistortPoints(
            pts.reshape(1, -1, 2).astype(np.float32),
            camera_matrix, dist_coeffs, P=camera_matrix
        ).reshape(4, 2)
        result[int(mid)] = pts_undist

    print(f"[top] 마커 검출: {sorted(result.keys())}")
    return result


def compute_top_homography(
    marker_corners: dict[int, np.ndarray],
    corner_ids: tuple[int, int, int, int],
    frame_px: tuple[float, float, float, float],
    output_size: tuple[int, int],
) -> np.ndarray | None:
    """
    4개 코너 마커 → LEGO BUILD AREA 정규화 호모그래피 계산.

    Parameters
    ----------
    marker_corners : detect_top_frame_markers() 반환값
    corner_ids     : (top-left, top-right, bottom-right, bottom-left) 마커 ID
                     setup.json의 "top_corner_ids" 값
    frame_px       : 각 코너 마커에서 사용할 픽셀 좌표 인덱스 (각 마커 코너 중 어느 것)
                     보통 마커의 내측 코너 1개를 사용
    output_size    : (width, height) 워핑 결과 해상도

    반환: 3×3 호모그래피 행렬, 실패 시 None
    """
    tl_id, tr_id, br_id, bl_id = corner_ids
    required = [tl_id, tr_id, br_id, bl_id]

    for mid in required:
        if mid not in marker_corners:
            print(f"[top] 코너 마커 {mid} 미검출 → 호모그래피 계산 불가")
            return None

    # 각 코너 마커의 내측 코너 픽셀 (마커 코너 순서: TL TR BR BL)
    # TL 마커 → 오른쪽 아래 코너(인덱스 2)가 BUILD AREA 코너에 가장 가깝다
    src_pts = np.array([
        marker_corners[tl_id][2],   # TL 마커의 BR 코너
        marker_corners[tr_id][3],   # TR 마커의 BL 코너
        marker_corners[br_id][0],   # BR 마커의 TL 코너
        marker_corners[bl_id][1],   # BL 마커의 TR 코너
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
        print("[top] 호모그래피 계산 실패")
        return None

    print(f"[top] 호모그래피 계산 완료 (인라이어: {mask.sum()}/4)")
    return H


def warp_build_area(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    호모그래피로 이미지를 변환해 LEGO BUILD AREA를 정면 뷰로 정렬한다.

    반환: warped BGR 이미지 (output_size 크기)
    """
    w, h = output_size
    warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)
    return warped
