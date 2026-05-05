"""
LEGO 색상 팔레트 및 양자화 유틸리티.

TOP 뷰 셀 기반 색상 추정은 top/grid_analyzer.py의 estimate_cell_color()에서 수행한다.
이 모듈은 LEGO 팔레트 정의와 snap_to_lego_color()만 제공한다.
"""

import numpy as np

# 공식 LEGO 색상 팔레트 (name → BGR)
LEGO_PALETTE: dict[str, tuple[int, int, int]] = {
    "Red":       (0,   0,   201),
    "Blue":      (152, 99,  11 ),
    "Yellow":    (0,   205, 248),
    "Green":     (0,   132, 70 ),
    "White":     (255, 255, 255),
    "Black":     (10,  10,  10 ),
    "Orange":    (0,   100, 226),
    "LightGray": (160, 165, 160),
    "DarkGray":  (100, 100, 100),
    "Tan":       (142, 188, 218),
    "DarkBlue":  (95,  57,  21 ),
    "Lime":      (60,  196, 163),
    "Pink":      (180, 112, 252),
    "Purple":    (120, 48,  130),
    "Brown":     (50,  80,  105),
}

_PALETTE_BGR   = np.array(list(LEGO_PALETTE.values()), dtype=np.float32)
_PALETTE_NAMES = list(LEGO_PALETTE.keys())


def snap_to_lego_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """BGR 색상을 가장 가까운 LEGO 팔레트 색으로 양자화한다."""
    q     = np.array(bgr, dtype=np.float32)
    dists = np.linalg.norm(_PALETTE_BGR - q, axis=1)
    idx   = int(np.argmin(dists))
    b, g, r = _PALETTE_BGR[idx].astype(int)
    return (int(b), int(g), int(r))


def color_name_from_bgr(bgr: tuple[int, int, int]) -> str:
    """BGR 색상에 해당하는 LEGO 팔레트 이름을 반환한다."""
    q     = np.array(bgr, dtype=np.float32)
    dists = np.linalg.norm(_PALETTE_BGR - q, axis=1)
    return _PALETTE_NAMES[int(np.argmin(dists))]
