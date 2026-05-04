"""
Per-view camera pose estimation using ChArUco.

Each view (TOP, FRONT, BACK, LEFT, RIGHT) has an image with a visible ChArUco
board.  This module returns the 4×4 extrinsic matrix that maps world coords
(board origin = LEGO base origin) into each camera's coordinate frame.

Convention: world origin = centre of ChArUco board on the LEGO base plate.
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
        """4×4 world-to-camera transform."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3,  3] = self.tvec.ravel()
        return T


def estimate_pose(image: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: np.ndarray,
                  view: str) -> CameraPose | None:
    """
    Estimate camera pose from a single view image.

    Returns CameraPose or None if detection failed.
    """
    rvec, tvec, corners, ids = detect_charuco(image, camera_matrix, dist_coeffs)
    if rvec is None:
        print(f"[pose] {view}: ChArUco detection failed")
        return None

    R, _ = cv2.Rodrigues(rvec)
    P    = camera_matrix @ np.hstack([R, tvec])

    print(f"[pose] {view}: t={tvec.ravel().round(4)}")
    return CameraPose(view=view, rvec=rvec, tvec=tvec, R=R, P=P)


def estimate_all_poses(images: dict[str, np.ndarray],
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray) -> dict[str, CameraPose]:
    """
    images : {view_name: bgr_image}  — provide whatever views are available
    Returns dict of successfully estimated poses.
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
