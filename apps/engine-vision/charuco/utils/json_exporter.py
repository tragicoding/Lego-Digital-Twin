"""
Python → Backend → Unity 전송용 JSON 빌더.

스키마
------
{
  "board": {
    "grid_width": 32,
    "grid_depth": 32,
    "max_height": 12,
    "unit": "stud"
  },
  "top_view": {
    "cells": [
      {"x": 0, "y": 0, "occupied": true, "color": "Red", "r": 201, "g": 0, "b": 0}
    ]
  },
  "side_views": {
    "FRONT": {"height_map": [0, 3, 5, ...]},
    "RIGHT": {"height_map": [0, 2, 4, ...]}
  },
  "voxels": [
    {"x": 0, "y": 0, "z": 0, "occupied": true, "color": "Red", "r": 201, "g": 0, "b": 0}
  ],
  "meta": {
    "timestamp": "...",
    "views_used": ["TOP", "FRONT", "RIGHT"],
    "total_voxels": 128,
    "method": "homography_grid"
  }
}
"""

import json
import datetime
from pathlib import Path

from ..top.grid_analyzer import CellResult
from ..reconstruction.voxel_grid import VoxelGrid


def build_json(
    top_cells:   list[CellResult],
    voxel_grid:  VoxelGrid,
    height_maps: dict[str, list],
    views_used:  list[str],
    grid_n:      int = 32,
    max_z:       int = 12,
) -> dict:
    """전체 JSON 페이로드를 빌드한다."""

    # top_view cells
    top_cells_data = [
        {"x": c.x, "y": c.y, "occupied": c.occupied,
         "color": c.color_name, "r": c.r, "g": c.g, "b": c.b}
        for c in top_cells
    ]

    # side_views height_maps
    side_views_data = {
        view: {"height_map": hmap.tolist() if hasattr(hmap, "tolist") else list(hmap)}
        for view, hmap in height_maps.items()
    }

    # (x,y) → 색상 매핑 (top_cells 기반)
    color_lookup = {
        (c.x, c.y): (c.color_name, c.r, c.g, c.b)
        for c in top_cells if c.occupied
    }

    # 점유된 복셀만 직렬화
    voxels_data = []
    for xi in range(voxel_grid.studs_x):
        for yi in range(voxel_grid.studs_y):
            for zi in range(voxel_grid.max_z):
                if voxel_grid.occupied[xi, yi, zi]:
                    color_name, r, g, b = color_lookup.get(
                        (xi, yi), ("LightGray", 160, 165, 160)
                    )
                    voxels_data.append({
                        "x": xi, "y": yi, "z": zi,
                        "occupied": True,
                        "color": color_name, "r": r, "g": g, "b": b,
                    })

    return {
        "board": {
            "grid_width": grid_n,
            "grid_depth": grid_n,
            "max_height": max_z,
            "unit":       "stud",
        },
        "top_view":   {"cells": top_cells_data},
        "side_views": side_views_data,
        "voxels":     voxels_data,
        "meta": {
            "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
            "views_used":   views_used,
            "total_voxels": len(voxels_data),
            "method":       "homography_grid",
        },
    }


def save_json(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[json] 저장 완료: 복셀 {payload['meta']['total_voxels']}개 → {path}")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
