"""
레고 복셀 그리드 데이터 구조.

좌표계 규약
-----------
  월드 원점 = LEGO BUILD AREA 중심
  X축       = 스터드 열 방향 (+오른쪽)
  Y축       = 스터드 행 방향 (+앞쪽)
  Z축       = 위쪽 (높이 방향)

복셀 인덱스 (xi, yi, zi) → 월드 좌표:
  x = (xi - cx) * STUD_PITCH
  y = (yi - cy) * STUD_PITCH
  z =  zi       * BRICK_HEIGHT

생성 방법:
  VoxelGrid는 직접 생성하지 않고 voxel_combiner.build_voxel_from_top_and_sides()로 만든다.
"""

import numpy as np
from dataclasses import dataclass, field

# 레고 물리 상수 (단위: 미터)
STUD_PITCH   = 0.008   # 스터드 피치 8 mm
BRICK_HEIGHT = 0.0096  # 브릭 높이 9.6 mm (플레이트 3개)
PLATE_HEIGHT = 0.0032  # 플레이트 높이 3.2 mm


@dataclass
class VoxelGrid:
    """레고 스터드 그리드에 정렬된 축-정렬 복셀 그리드."""

    studs_x: int = 32    # 가로 스터드 수 (grid_n과 동일하게 설정)
    studs_y: int = 32    # 세로 스터드 수
    max_z:   int = 12    # 최대 브릭 층수
    z_step:  float = BRICK_HEIGHT

    occupied: np.ndarray = field(init=False)  # bool (studs_x, studs_y, max_z)

    def __post_init__(self):
        self.occupied = np.zeros(
            (self.studs_x, self.studs_y, self.max_z), dtype=bool
        )

    def voxel_center_world(self, xi: int, yi: int, zi: int) -> np.ndarray:
        """복셀 중심의 월드 좌표 (X, Y, Z) [미터] 를 반환한다."""
        cx = self.studs_x / 2.0
        cy = self.studs_y / 2.0
        x  = (xi - cx + 0.5) * STUD_PITCH
        y  = (yi - cy + 0.5) * STUD_PITCH
        z  = (zi + 0.5)      * self.z_step
        return np.array([x, y, z], dtype=np.float64)

    @property
    def occupied_count(self) -> int:
        return int(self.occupied.sum())

    def to_list(self) -> list[dict]:
        """점유된 복셀을 JSON 직렬화용 dict 리스트로 반환한다."""
        result = []
        for xi in range(self.studs_x):
            for yi in range(self.studs_y):
                for zi in range(self.max_z):
                    if self.occupied[xi, yi, zi]:
                        result.append({"x": xi, "y": yi, "z": zi})
        return result
