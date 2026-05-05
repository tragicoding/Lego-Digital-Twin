"""
Multi-view Space Carving 오케스트레이터.

알고리즘 (Visual Hull / Space Carving)
---------------------------------------
1. 복셀 그리드를 모두 채워진 상태로 초기화한다.
2. 각 카메라 뷰에 대해:
   a. 점유된 복셀을 이미지 평면에 투영한다.
   b. 실루엣 마스크 밖으로 투영되는 복셀을 제거(carve)한다.
3. 남은 점유 복셀이 Visual Hull 근사 결과가 된다.

의사 코드
----------
grid = VoxelGrid(studs_x=16, studs_y=16, max_z=10)  # 전체 True

for view in [TOP, FRONT, BACK, LEFT, RIGHT]:
    mask = silhouettes[view]
    pose = poses[view]
    for every voxel v in grid (occupied only):
        p = project(v.world_center, pose.P, K, D)
        if p is inside image AND mask[p] == 0:
            grid[v] = False   # 제거

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
    Multi-view Space Carving을 실행한다.

    Parameters
    ----------
    grid       : VoxelGrid (in-place 수정됨)
    silhouettes: {뷰 이름: SilhouetteResult}
    poses      : {뷰 이름: CameraPose}
    iterations : 수렴까지 반복 패스 수
    """
    grid.reset()

    for it in range(iterations):
        total_carved = 0
        for view, sil in silhouettes.items():
            if view not in poses:
                print(f"[carver] {view} 포즈 없음, 스킵")
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
            break  # 수렴: 더 이상 제거할 복셀 없음

    print(f"[carver] 최종 점유 복셀 수: {grid.occupied_count}")
    return grid


def apply_vertical_floor_constraint(grid: VoxelGrid) -> None:
    """
    부유 복셀 제거: (x,y,z>0) 위치의 복셀은 아래 (x,y,z-1)이 점유된 경우에만 유효하다.
    아래→위 방향으로 한 번 전파한다.
    """
    for zi in range(1, grid.max_z):
        floaters = grid.occupied[:, :, zi] & ~grid.occupied[:, :, zi - 1]
        grid.occupied[:, :, zi][floaters] = False
