"""
TOP 점유맵 + SIDE 높이맵 → 3D 복셀 조합 모듈.

알고리즘
--------
1단계 (TOP only):
  top_cells에서 occupied=True인 (x, y) → voxel[x][y][0] = True (단층)

2단계 (TOP + 1 side):
  FRONT 또는 RIGHT 높이맵으로 z_max 결정
  voxel[x][y][0 ~ z_max] = True

3단계 (TOP + FRONT/BACK + LEFT/RIGHT):
  h_xz[x] = max(FRONT[x], BACK[x])    ← x방향 높이 (z축)
  h_yz[y] = max(LEFT[y], RIGHT[y])     ← y방향 높이 (z축)
  z_max[x,y] = min(h_xz[x], h_yz[y])  ← 교집합

pseudocode:
  voxel = zeros(grid_n, grid_n, max_z)
  for cell in top_cells where occupied:
      x, y = cell.x, cell.y
      z_max = min(h_xz.get(x, max_z), h_yz.get(y, max_z))
      voxel[x][y][0 : z_max] = True
"""

import numpy as np
from .voxel_grid import VoxelGrid
from ..top.grid_analyzer import CellResult


def build_voxel_from_top_and_sides(
    top_cells: list[CellResult],
    height_maps: dict[str, np.ndarray],
    grid_n: int = 32,
    max_z: int  = 12,
) -> VoxelGrid:
    """
    TOP 점유맵과 SIDE 높이맵을 조합해 3D VoxelGrid를 생성한다.

    Parameters
    ----------
    top_cells    : generate_xy_grid() 반환 리스트
    height_maps  : {뷰 이름: height_map (grid_n,)}
                   지원 키: "FRONT", "BACK", "LEFT", "RIGHT"
    grid_n       : 격자 분할 수 (top_cells과 동일해야 함)
    max_z        : 최대 높이 레이어 수

    반환: VoxelGrid (occupied 배열이 채워진 상태)
    """
    grid = VoxelGrid(studs_x=grid_n, studs_y=grid_n, max_z=max_z)

    # x방향 높이: FRONT/BACK 중 더 높은 값 사용
    h_xz = _merge_height_maps(height_maps, ["FRONT", "BACK"], grid_n, max_z)
    # y방향 높이: LEFT/RIGHT 중 더 높은 값 사용
    h_yz = _merge_height_maps(height_maps, ["LEFT", "RIGHT"], grid_n, max_z)

    for cell in top_cells:
        if not cell.occupied:
            continue
        x, y = cell.x, cell.y

        # 교집합: 두 방향 모두 만족하는 최소 높이
        z_max = min(h_xz[x], h_yz[y])
        z_max = max(1, min(z_max, max_z))  # 최소 1층 보장

        grid.occupied[x, y, 0:z_max] = True

    apply_floor_constraint(grid)

    print(f"[combiner] 3D 복셀 생성 완료: 총 {grid.occupied_count}개")
    return grid


def _merge_height_maps(
    height_maps: dict[str, np.ndarray],
    keys: list[str],
    grid_n: int,
    max_z: int,
) -> np.ndarray:
    """
    여러 높이맵 중 이용 가능한 것의 최댓값을 사용한다.
    해당 방향 맵이 없으면 max_z(최대 높이)로 채운다 → 제약 없음.
    """
    result = np.full(grid_n, max_z, dtype=np.int32)
    found  = False
    for key in keys:
        if key in height_maps:
            result = np.maximum(result if found else np.zeros(grid_n, dtype=np.int32),
                                height_maps[key])
            found = True
    return result


def apply_floor_constraint(grid: VoxelGrid) -> None:
    """
    부유 복셀 제거: (x,y,z>0) 위치의 복셀은 아래 (x,y,z-1)이 점유돼야 유효하다.
    아래→위 방향으로 한 번 전파한다.
    """
    for zi in range(1, grid.max_z):
        floaters = grid.occupied[:, :, zi] & ~grid.occupied[:, :, zi - 1]
        grid.occupied[:, :, zi][floaters] = False
