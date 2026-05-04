"""
ChArUco 보드 생성 및 검출 모듈.

물리적 배치 권장사항
--------------------
  - 보드 1장을 레고 베이스 위에 평평하게 배치 (TOP 뷰 기준 원점)
  - 나머지 4장을 Front / Back / Left / Right 벽면에 각각 배치
  - 보드마다 고유 ArUco ID 범위가 달라 시점 구분이 가능하다
  - 일부 마커가 가려져도 코너 기반 추정이 가능한 것이 ChArUco의 장점
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os

# 보드 파라미터 (인쇄 크기에 맞게 조정, 단위: 미터)
SQUARES_X   = 7      # 열 개수
SQUARES_Y   = 5      # 행 개수
SQUARE_LEN  = 0.03   # 정사각형 한 변 (30 mm)
MARKER_LEN  = 0.022  # ArUco 마커 한 변 (22 mm)

ARUCO_DICT  = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
BOARD       = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, ARUCO_DICT)


def generate_board_image(save_path: str, px_per_square: int = 100) -> None:
    """보드 이미지를 PNG로 저장한다 (인쇄용)."""
    w = SQUARES_X * px_per_square
    h = SQUARES_Y * px_per_square
    img = BOARD.generateImage((w, h), marginSize=20, borderBits=1)
    cv2.imwrite(save_path, img)
    print(f"[charuco] 보드 이미지 저장 → {save_path}")


def detect_charuco(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    ChArUco 코너를 검출하고 보드 포즈를 추정한다.

    반환값
    ------
    rvec, tvec : 회전·평행이동 벡터 (보드→카메라), 실패 시 (None, None)
    corners    : 서브픽셀 코너 좌표
    ids        : 코너 ID
    """
    detector_params = aruco.DetectorParameters()
    detector        = aruco.ArucoDetector(ARUCO_DICT, detector_params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) < 4:
        return None, None, None, None

    charuco_detector = aruco.CharucoDetector(BOARD)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < 6:
        return None, None, charuco_corners, charuco_ids

    ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, BOARD, camera_matrix, dist_coeffs, None, None
    )

    if not ok:
        return None, None, charuco_corners, charuco_ids

    return rvec, tvec, charuco_corners, charuco_ids


def draw_axis(image: np.ndarray, rvec, tvec,
              camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
              length: float = 0.05) -> np.ndarray:
    """좌표축을 이미지에 오버레이해 검출 결과를 시각적으로 확인한다."""
    return cv2.drawFrameAxes(image.copy(), camera_matrix, dist_coeffs,
                             rvec, tvec, length)
