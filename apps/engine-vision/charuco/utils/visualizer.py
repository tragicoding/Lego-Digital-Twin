"""
Debug visualisation utilities.

- draw_voxel_grid_top_view : 2D top-down occupancy map
- draw_silhouette_overlay  : mask overlaid on image
- open3d_preview           : interactive 3D voxel viewer
"""

import cv2
import numpy as np


def draw_voxel_grid_top_view(occupied: np.ndarray,
                              cell_px: int = 20) -> np.ndarray:
    """
    occupied : bool array (studs_x, studs_y, max_z)
    Returns  : BGR image showing top-down XY footprint.
    Colors indicate height (z) of tallest occupied voxel per column.
    """
    sx, sy, sz = occupied.shape
    img = np.ones((sy * cell_px, sx * cell_px, 3), dtype=np.uint8) * 240

    for xi in range(sx):
        for yi in range(sy):
            col = occupied[xi, yi, :]
            if col.any():
                z_top = int(col.nonzero()[0][-1])
                hue   = int(120 - z_top * (120 / max(sz - 1, 1)))  # green→red with height
                color = cv2.cvtColor(
                    np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
                x0 = xi * cell_px
                y0 = yi * cell_px
                cv2.rectangle(img, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1),
                              color, -1)
                cv2.rectangle(img, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1),
                              (0, 0, 0), 1)
    return img


def draw_silhouette_overlay(image: np.ndarray, mask: np.ndarray,
                             alpha: float = 0.4) -> np.ndarray:
    overlay = image.copy()
    green   = np.zeros_like(image)
    green[:, :, 1] = 255
    overlay[mask > 0] = cv2.addWeighted(
        image[mask > 0], 1 - alpha,
        green[mask > 0], alpha, 0
    )
    return overlay


def open3d_preview(occupied: np.ndarray, voxel_grid) -> None:
    """Interactive Open3D voxel viewer."""
    try:
        import open3d as o3d
    except ImportError:
        print("[viz] open3d not installed; skipping 3D preview")
        return

    pts = []
    for xi in range(occupied.shape[0]):
        for yi in range(occupied.shape[1]):
            for zi in range(occupied.shape[2]):
                if occupied[xi, yi, zi]:
                    pts.append(voxel_grid.voxel_center_world(xi, yi, zi))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    o3d.visualization.draw_geometries([pcd], window_name="LEGO Voxel Preview")
