"""
ChArUco 기반 뷰별 카메라 포즈 추정 모듈.

TOP / FRONT / BACK / LEFT / RIGHT 각 뷰 이미지에서
ChArUco 보드를 검출해 카메라 외부 파라미터(extrinsic)를 계산한다.

좌표계 규약
-----------
  월드 원점 = 레고 베이스 플레이트 위 ChArUco 보드 중심
"""

import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

from .charuco_board import detect_charuco


VIEW_NAMES = ("TOP", "FRONT", "BACK", "LEFT", "RIGHT")


@dataclass
class CameraPose:
    view: str
    rvec: np.ndarray   # (3,1) Rodrigues rotation
    tvec: np.ndarray   # (3,1) translation  [metres]
    R: np.ndarray      # (3,3) rotation matrix
    P: np.ndarray      # (3,4) projection matrix  [K | 0] · [R | t]

    @property
    def extrinsic(self) -> np.ndarray:
        """4×4 월드→카메라 변환 행렬."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3,  3] = self.tvec.ravel()
        return T


def estimate_pose(image: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: np.ndarray,
                  view: str) -> CameraPose | None:
    """단일 뷰 이미지에서 카메라 포즈를 추정한다. 실패 시 None 반환."""
    rvec, tvec, corners, ids = detect_charuco(image, camera_matrix, dist_coeffs)
    if rvec is None:
        print(f"[pose] {view}: ChArUco 검출 실패")
        return None

    R, _ = cv2.Rodrigues(rvec)
    P    = camera_matrix @ np.hstack([R, tvec])

    print(f"[pose] {view}: t={tvec.ravel().round(4)}")
    return CameraPose(view=view, rvec=rvec, tvec=tvec, R=R, P=P)


def estimate_all_poses(images: dict[str, np.ndarray],
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray) -> dict[str, CameraPose]:
    """
    images : {view_name: bgr_image} — 가용한 뷰만 전달해도 된다.
    성공적으로 추정된 포즈 dict를 반환한다.
    """
    poses = {}
    for view, img in images.items():
        pose = estimate_pose(img, camera_matrix, dist_coeffs, view)
        if pose is not None:
            poses[view] = pose
    return poses


def save_poses(path: str, poses: dict[str, CameraPose]) -> None:
    data = {}
    for view, p in poses.items():
        data[f"{view}_rvec"]   = p.rvec
        data[f"{view}_tvec"]   = p.tvec
        data[f"{view}_R"]      = p.R
        data[f"{view}_P"]      = p.P
    np.savez(path, **data)
    print(f"[pose] Saved {len(poses)} poses → {path}")


def load_poses(path: str,
               camera_matrix: np.ndarray,
               dist_coeffs: np.ndarray) -> dict[str, CameraPose]:
    data  = np.load(path)
    views = {k.replace("_rvec", "") for k in data.files if k.endswith("_rvec")}
    poses = {}
    for view in views:
        rvec = data[f"{view}_rvec"]
        tvec = data[f"{view}_tvec"]
        R    = data[f"{view}_R"]
        P    = data[f"{view}_P"]
        poses[view] = CameraPose(view=view, rvec=rvec, tvec=tvec, R=R, P=P)
    return poses
