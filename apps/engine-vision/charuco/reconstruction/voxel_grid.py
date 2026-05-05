"""
레고 복셀 그리드 정의 및 Multi-view Space Carving 모듈.

좌표계 규약
-----------
  월드 원점 = 레고 베이스 플레이트 위 ChArUco 보드 중심
  X축       = 스터드 열 방향 (+오른쪽)
  Y축       = 스터드 행 방향 (+앞쪽)
  Z축       = 위쪽 (높이 방향)

복셀 인덱스 (xi, yi, zi) → 월드 좌표:
  x = (xi - cx) * STUD_PITCH
  y = (yi - cy) * STUD_PITCH
  z =  zi       * BRICK_HEIGHT
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

    studs_x: int = 16    # 베이스 플레이트 가로 스터드 수 (16×16 기본)
    studs_y: int = 16    # 베이스 플레이트 세로 스터드 수
    max_z:   int = 10    # max brick layers  (10 bricks tall)
    z_step:  float = BRICK_HEIGHT

    # 내부 상태 (__post_init__ 에서 초기화)
    occupied: np.ndarray = field(init=False)  # bool  (x, y, z)
    vote:     np.ndarray = field(init=False)  # int16 (x, y, z) — 카빙 투표 수

    def __post_init__(self):
        shape = (self.studs_x, self.studs_y, self.max_z)
        self.occupied = np.ones(shape, dtype=bool)    # space carving: 처음엔 모두 채워진 상태
        self.vote     = np.zeros(shape, dtype=np.int16)

    # ------------------------------------------------------------------
    # 좌표 변환 헬퍼
    # ------------------------------------------------------------------

    def voxel_center_world(self, xi: int, yi: int, zi: int) -> np.ndarray:
        """복셀 중심의 월드 좌표 (X, Y, Z) [미터] 를 반환한다."""
        cx = self.studs_x / 2.0
        cy = self.studs_y / 2.0
        x  = (xi - cx + 0.5) * STUD_PITCH
        y  = (yi - cy + 0.5) * STUD_PITCH
        z  = (zi + 0.5)      * self.z_step
        return np.array([x, y, z], dtype=np.float64)

    def all_voxel_centers(self) -> np.ndarray:
        """모든 복셀 중심의 월드 좌표 배열 (N, 3)을 반환한다."""
        sx, sy, sz = self.studs_x, self.studs_y, self.max_z
        xi = np.arange(sx)
        yi = np.arange(sy)
        zi = np.arange(sz)
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')   # (sx,sy,sz)

        cx, cy = sx / 2.0, sy / 2.0
        X = (gx - cx + 0.5) * STUD_PITCH
        Y = (gy - cy + 0.5) * STUD_PITCH
        Z = (gz + 0.5)      * self.z_step
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        return pts   # (sx*sy*sz, 3)

    # ------------------------------------------------------------------
    # Space Carving
    # ------------------------------------------------------------------

    def carve_with_mask(self, projection_matrix: np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray,
                         silhouette_mask: np.ndarray,
                         rvec: np.ndarray,
                         tvec: np.ndarray) -> int:
        """
        하나의 카메라 뷰로 점유된 복셀을 투영해
        실루엣 마스크 밖에 있는 복셀을 제거(carve)한다.

        반환값: 이번 패스에서 제거된 복셀 수
        """
        pts_world = self.all_voxel_centers()             # (N,3)
        indices   = np.array(list(np.ndindex(
            self.studs_x, self.studs_y, self.max_z)))    # (N,3)  xi,yi,zi

        # OpenCV projectPoints: 렌즈 왜곡 포함 투영
        pts_img, _ = cv2.projectPoints(
            pts_world.reshape(-1, 1, 3).astype(np.float32),
            rvec, tvec, camera_matrix, dist_coeffs
        )  # (N,1,2)
        pts_img = pts_img.reshape(-1, 2)

        h, w = silhouette_mask.shape[:2]
        carved = 0
        for n, (u, v) in enumerate(pts_img):
            xi, yi, zi = indices[n]
            if not self.occupied[xi, yi, zi]:
                continue
            iu, iv = int(round(u)), int(round(v))
            # 프레임 밖 → 보수적으로 처리 (카메라 뒤쪽 복셀일 수 있으므로 제거하지 않음)
            if 0 <= iu < w and 0 <= iv < h:
                if silhouette_mask[iv, iu] == 0:
                    self.occupied[xi, yi, zi] = False
                    carved += 1
        return carved

    def reset(self):
        self.occupied[:] = True
        self.vote[:]     = 0

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


# Avoid circular import from carve_with_mask
import cv2
