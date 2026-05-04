"""
LEGO voxel grid definition and multi-view carving.

Coordinate system
-----------------
World origin  = ChArUco board centre on LEGO base plate
X axis        = LEGO stud column direction  (+right)
Y axis        = LEGO stud row direction     (+forward)
Z axis        = up  (stud height direction)

LEGO stud pitch = 8 mm → voxel side = STUD_PITCH metres
One LEGO plate  = 3.2 mm tall; one LEGO brick = 9.6 mm (3 plates)
→ z_step = BRICK_HEIGHT / z_subdivisions

Voxel index (xi, yi, zi) maps to world position:
  x = (xi - cx) * STUD_PITCH
  y = (yi - cy) * STUD_PITCH
  z =  zi       * Z_STEP
"""

import numpy as np
from dataclasses import dataclass, field

# Physical LEGO constants (metres)
STUD_PITCH   = 0.008   # 8 mm
BRICK_HEIGHT = 0.0096  # 9.6 mm per full brick height
PLATE_HEIGHT = 0.0032  # 3.2 mm per plate


@dataclass
class VoxelGrid:
    """Axis-aligned voxel grid aligned to LEGO stud grid."""

    studs_x: int = 16    # baseplate width  (16×16 standard)
    studs_y: int = 16    # baseplate depth
    max_z:   int = 10    # max brick layers  (10 bricks tall)
    z_step:  float = BRICK_HEIGHT

    # Internal state (set in __post_init__)
    occupied: np.ndarray = field(init=False)  # bool  (x, y, z)
    vote:     np.ndarray = field(init=False)  # int16 (x, y, z) — carving votes

    def __post_init__(self):
        shape = (self.studs_x, self.studs_y, self.max_z)
        self.occupied = np.ones(shape, dtype=bool)   # start full (space carving)
        self.vote     = np.zeros(shape, dtype=np.int16)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def voxel_center_world(self, xi: int, yi: int, zi: int) -> np.ndarray:
        """Return (X, Y, Z) world position of voxel centre [metres]."""
        cx = self.studs_x / 2.0
        cy = self.studs_y / 2.0
        x  = (xi - cx + 0.5) * STUD_PITCH
        y  = (yi - cy + 0.5) * STUD_PITCH
        z  = (zi + 0.5)      * self.z_step
        return np.array([x, y, z], dtype=np.float64)

    def all_voxel_centers(self) -> np.ndarray:
        """Return (N, 3) array of all voxel centres in world coords."""
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
    # Space carving
    # ------------------------------------------------------------------

    def carve_with_mask(self, projection_matrix: np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray,
                         silhouette_mask: np.ndarray,
                         rvec: np.ndarray,
                         tvec: np.ndarray) -> int:
        """
        Project all occupied voxels into one camera view.
        Remove (carve) voxels that project OUTSIDE the silhouette.

        Returns number of voxels carved this pass.
        """
        pts_world = self.all_voxel_centers()             # (N,3)
        indices   = np.array(list(np.ndindex(
            self.studs_x, self.studs_y, self.max_z)))    # (N,3)  xi,yi,zi

        # Project to image plane using OpenCV (handles distortion)
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
            # Out-of-frame → conservative: do NOT carve (voxel may be behind camera)
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
        """Export occupied voxels as list of dicts for JSON serialisation."""
        result = []
        for xi in range(self.studs_x):
            for yi in range(self.studs_y):
                for zi in range(self.max_z):
                    if self.occupied[xi, yi, zi]:
                        result.append({"x": xi, "y": yi, "z": zi})
        return result


# Avoid circular import from carve_with_mask
import cv2
