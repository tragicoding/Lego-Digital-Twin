"""
ChArUco board generation and detection.

Board layout (physical):
  - Place one board flat on the LEGO base (top-facing, identifies board origin)
  - Place four boards on each side wall (front, back, left, right)
  - Each board has a unique BOARD_ID so poses can be distinguished per view
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Board parameters (tune to match printed size)
SQUARES_X   = 7    # checkerboard columns
SQUARES_Y   = 5    # checkerboard rows
SQUARE_LEN  = 0.03 # metres per square (30 mm)
MARKER_LEN  = 0.022 # metres per ArUco marker (22 mm)

ARUCO_DICT  = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
BOARD       = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, ARUCO_DICT)


def generate_board_image(save_path: str, px_per_square: int = 100) -> None:
    """Render the board to a PNG file for printing."""
    w = SQUARES_X * px_per_square
    h = SQUARES_Y * px_per_square
    img = BOARD.generateImage((w, h), marginSize=20, borderBits=1)
    cv2.imwrite(save_path, img)
    print(f"[charuco] Board image saved → {save_path}")


def detect_charuco(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    Detect ChArUco corners and estimate board pose.

    Returns
    -------
    rvec, tvec : rotation and translation vectors (board → camera), or (None, None)
    corners    : refined sub-pixel corner positions
    ids        : corner IDs
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
    """Overlay coordinate axes on image for visual verification."""
    return cv2.drawFrameAxes(image.copy(), camera_matrix, dist_coeffs,
                             rvec, tvec, length)
