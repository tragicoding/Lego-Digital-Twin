"""
ChArUco 이미지 세트를 이용한 카메라 내부 파라미터(intrinsic) 보정 모듈.

실행 방법
---------
  python -m charuco.calibration.camera_calibrator \\
      --images data/calib/*.jpg \\
      --out    data/camera_params.npz
"""

import argparse
import glob
import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path

from .charuco_board import ARUCO_DICT, BOARD


def calibrate_from_images(image_paths: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    ChArUco 기반 내부 파라미터 보정을 실행한다.

    반환값
    ------
    camera_matrix, dist_coeffs, rms_error
    """
    all_corners, all_ids = [], []
    img_size = None

    charuco_detector = aruco.CharucoDetector(BOARD)
    aruco_detector   = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())

    for path in image_paths:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        corners, ids, _, _ = charuco_detector.detectBoard(gray)
        if corners is not None and len(corners) >= 6:
            all_corners.append(corners)
            all_ids.append(ids)

    if len(all_corners) < 5:
        raise ValueError(f"Need ≥5 valid frames, got {len(all_corners)}")

    rms, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(
        all_corners, all_ids, BOARD, img_size, None, None
    )
    print(f"[calibration] RMS reprojection error: {rms:.4f} px  ({len(all_corners)} frames)")
    return camera_matrix, dist_coeffs, rms


def save_params(path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
    np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"[calibration] Saved → {path}")


def load_params(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="glob pattern e.g. data/calib/*.jpg")
    parser.add_argument("--out",    default="data/camera_params.npz")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images))
    K, D, rms = calibrate_from_images(paths)
    save_params(args.out, K, D)
