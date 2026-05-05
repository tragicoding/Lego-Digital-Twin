"""
JSON schema for Python → Unity voxel transfer.

Schema
------
{
  "version": "1.0",
  "timestamp": "2026-05-05T14:23:00Z",
  "board": {
    "studs_x": 16,
    "studs_y": 16,
    "stud_pitch_mm": 8.0,
    "brick_height_mm": 9.6
  },
  "voxels": [
    {
      "x": 3, "y": 4, "z": 0,
      "r": 201, "g": 0, "b": 0,
      "color_name": "Red"
    },
    ...
  ],
  "meta": {
    "total_voxels": 128,
    "views_used": ["TOP","FRONT","RIGHT"],
    "method": "multi_view_space_carving"
  }
}

Unity reads this via LegoVoxelLoader.cs and spawns procedural blocks.
"""

import json
import datetime
from pathlib import Path

from ..reconstruction.voxel_grid import STUD_PITCH, BRICK_HEIGHT


def build_json(voxels_with_color: list[dict],
               studs_x: int,
               studs_y: int,
               views_used: list[str],
               extra_meta: dict | None = None) -> dict:
    """Build the full payload dict."""
    payload = {
        "version": "1.0",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "board": {
            "studs_x":        studs_x,
            "studs_y":        studs_y,
            "stud_pitch_mm":  STUD_PITCH * 1000,
            "brick_height_mm": BRICK_HEIGHT * 1000,
        },
        "voxels": voxels_with_color,
        "meta": {
            "total_voxels": len(voxels_with_color),
            "views_used":   views_used,
            "method":       "multi_view_space_carving",
            **(extra_meta or {}),
        }
    }
    return payload


def save_json(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[json] Saved {len(payload['voxels'])} voxels → {path}")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
