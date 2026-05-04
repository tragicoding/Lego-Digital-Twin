"""
Multi-view space carving orchestrator.

Algorithm (Visual Hull / Space Carving)
----------------------------------------
1. Initialise voxel grid as fully occupied.
2. For each camera view:
   a. Project all occupied voxels onto image plane.
   b. Voxels that fall OUTSIDE the silhouette mask are carved away.
3. Remaining occupied voxels form the Visual Hull approximation.
4. Optionally apply a minimum-vote threshold to reduce noise.

Pseudocode
----------
grid = VoxelGrid(studs_x=16, studs_y=16, max_z=10)  # all True

for view in [TOP, FRONT, BACK, LEFT, RIGHT]:
    mask = silhouettes[view]
    pose = poses[view]
    for every voxel v in grid (occupied only):
        p = project(v.world_center, pose.P, K, D)
        if p is inside image AND mask[p] == 0:
            grid[v] = False   # carved

result = [v for v in grid if v.occupied]
"""

from __future__ import annotations
import numpy as np
from .voxel_grid import VoxelGrid
from .silhouette import SilhouetteResult
from ..calibration.pose_estimator import CameraPose


def run_space_carving(grid: VoxelGrid,
                      silhouettes: dict[str, SilhouetteResult],
                      poses:       dict[str, CameraPose],
                      camera_matrix: np.ndarray,
                      dist_coeffs:   np.ndarray,
                      iterations: int = 2) -> VoxelGrid:
    """
    Run multi-view space carving.

    Parameters
    ----------
    grid       : VoxelGrid (will be modified in-place)
    silhouettes: {view_name: SilhouetteResult}
    poses      : {view_name: CameraPose}
    iterations : repeat passes to catch newly freed voxels
    """
    grid.reset()

    for it in range(iterations):
        total_carved = 0
        for view, sil in silhouettes.items():
            if view not in poses:
                print(f"[carver] no pose for {view}, skipping")
                continue
            pose   = poses[view]
            carved = grid.carve_with_mask(
                projection_matrix = pose.P,
                camera_matrix     = camera_matrix,
                dist_coeffs       = dist_coeffs,
                silhouette_mask   = sil.mask,
                rvec              = pose.rvec,
                tvec              = pose.tvec,
            )
            total_carved += carved
            print(f"[carver] iter={it} view={view}: carved {carved} voxels "
                  f"(remaining {grid.occupied_count})")

        if total_carved == 0:
            break  # converged

    print(f"[carver] Final occupied voxels: {grid.occupied_count}")
    return grid


def apply_vertical_floor_constraint(grid: VoxelGrid) -> None:
    """
    Remove floating voxels: a voxel at (x,y,z>0) is only valid if
    (x,y,z-1) is also occupied.  Propagate downward once.
    """
    for zi in range(1, grid.max_z):
        floaters = grid.occupied[:, :, zi] & ~grid.occupied[:, :, zi - 1]
        grid.occupied[:, :, zi][floaters] = False
