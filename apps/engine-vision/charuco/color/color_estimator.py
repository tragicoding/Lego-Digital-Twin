"""
Per-voxel colour estimation.

Strategy
--------
Primary  : TOP view  → sample pixel at projected (xi, yi) position.
Fallback : FRONT/RIGHT side views for voxels not visible from top.

Each occupied voxel gets an RGB colour assigned by projecting its world
centre into the most reliable view and sampling the (undistorted) image.

LEGO colour quantisation (optional)
-------------------------------------
After raw colour sampling, snap to the nearest official LEGO colour using
the provided palette to improve realism in Unity.
"""

import cv2
import numpy as np

# Official LEGO colour palette (name → BGR)
LEGO_PALETTE: dict[str, tuple[int, int, int]] = {
    "Red":         (0,   0,   201),
    "Blue":        (152, 99,  11),
    "Yellow":      (0,   205, 248),
    "Green":       (0,   132, 70),
    "White":       (255, 255, 255),
    "Black":       (10,  10,  10),
    "Orange":      (0,   100, 226),
    "LightGray":   (160, 165, 160),
    "DarkGray":    (100, 100, 100),
    "Tan":         (142, 188, 218),
    "DarkBlue":    (95,  57,  21),
    "Lime":        (60,  196, 163),
    "Pink":        (180, 112, 252),
    "Purple":      (120, 48,  130),
    "Brown":       (50,  80,  105),
}

_PALETTE_BGR  = np.array(list(LEGO_PALETTE.values()), dtype=np.float32)
_PALETTE_NAMES = list(LEGO_PALETTE.keys())


def snap_to_lego_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return nearest LEGO palette colour in BGR."""
    q     = np.array(bgr, dtype=np.float32)
    dists = np.linalg.norm(_PALETTE_BGR - q, axis=1)
    idx   = int(np.argmin(dists))
    r, g, b = _PALETTE_BGR[idx].astype(int)
    return (int(r), int(g), int(b))


def sample_color_at_voxel(voxel_world: np.ndarray,
                           image: np.ndarray,
                           rvec: np.ndarray,
                           tvec: np.ndarray,
                           camera_matrix: np.ndarray,
                           dist_coeffs: np.ndarray) -> tuple[int, int, int] | None:
    """
    Project voxel_world (3,) into image and return sampled BGR pixel.
    Returns None if projection is outside image bounds.
    """
    pt = voxel_world.reshape(1, 1, 3).astype(np.float32)
    uv, _ = cv2.projectPoints(pt, rvec, tvec, camera_matrix, dist_coeffs)
    u, v  = int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))
    h, w  = image.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        b, g, r = image[v, u].astype(int)
        return (b, g, r)
    return None


def estimate_colors(occupied_voxels: list[dict],
                    voxel_grid,
                    images: dict[str, np.ndarray],
                    poses:  dict,
                    camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray,
                    quantise: bool = True,
                    view_priority: list[str] = ("TOP", "FRONT", "RIGHT")) -> list[dict]:
    """
    Assign colour to each occupied voxel.

    Returns voxel list with added "r","g","b" fields and "color_name".
    """
    result = []
    for v in occupied_voxels:
        xi, yi, zi = v["x"], v["y"], v["z"]
        world_pt   = voxel_grid.voxel_center_world(xi, yi, zi)

        color_bgr = None
        for view in view_priority:
            if view not in images or view not in poses:
                continue
            color_bgr = sample_color_at_voxel(
                world_pt, images[view],
                poses[view].rvec, poses[view].tvec,
                camera_matrix, dist_coeffs
            )
            if color_bgr is not None:
                break

        if color_bgr is None:
            color_bgr = (128, 128, 128)  # default grey

        if quantise:
            color_bgr = snap_to_lego_color(color_bgr)

        b, g, r = color_bgr
        color_name = _PALETTE_NAMES[
            int(np.argmin(np.linalg.norm(_PALETTE_BGR - np.array([b, g, r], dtype=np.float32), axis=1)))
        ]

        result.append({**v, "r": r, "g": g, "b": b, "color_name": color_name})

    return result
