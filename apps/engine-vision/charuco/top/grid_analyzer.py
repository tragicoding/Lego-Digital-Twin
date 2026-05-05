"""
TOP 뷰 워핑 이미지 → N×N 격자 분석 모듈.

각 셀에 대해:
  - occupied : 레고 점유 여부
  - color    : LEGO 팔레트 양자화 색상

배경 이미지(BG_TOP)가 있으면 차분으로 점유 판단.
없으면 채도(HSV-S) 기반으로 판단.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from ..color.color_estimator import snap_to_lego_color, LEGO_PALETTE


@dataclass
class CellResult:
    x: int
    y: int
    occupied: bool
    color_name: str
    r: int
    g: int
    b: int


def generate_xy_grid(
    warped: np.ndarray,
    grid_n: int = 32,
    bg_warped: np.ndarray | None = None,
    occupancy_threshold: int = 25,
    saturation_threshold: int = 40,
) -> list[CellResult]:
    """
    워핑된 TOP 이미지를 grid_n × grid_n 셀로 분할하고 각 셀을 분석한다.

    Parameters
    ----------
    warped               : warp_build_area() 반환 이미지
    grid_n               : 격자 분할 수 (기본 32 → 32×32 = 1024셀)
    bg_warped            : 레고 없는 배경 워핑 이미지 (있으면 차분 방식)
    occupancy_threshold  : 배경 차분 픽셀 차이 임계값
    saturation_threshold : 채도 기반 점유 판단 임계값 (bg_warped 없을 때)

    반환: CellResult 리스트 (grid_n² 개)
    """
    h, w = warped.shape[:2]
    cell_h = h / grid_n
    cell_w = w / grid_n

    results = []
    for yi in range(grid_n):
        for xi in range(grid_n):
            y0 = int(yi * cell_h)
            y1 = int((yi + 1) * cell_h)
            x0 = int(xi * cell_w)
            x1 = int((xi + 1) * cell_w)

            cell     = warped[y0:y1, x0:x1]
            bg_cell  = bg_warped[y0:y1, x0:x1] if bg_warped is not None else None

            occupied   = analyze_cell_occupancy(cell, bg_cell, occupancy_threshold, saturation_threshold)
            color_name, r, g, b = estimate_cell_color(cell) if occupied else ("Empty", 0, 0, 0)

            results.append(CellResult(
                x=xi, y=yi,
                occupied=occupied,
                color_name=color_name,
                r=r, g=g, b=b,
            ))

    occupied_count = sum(1 for c in results if c.occupied)
    print(f"[top] 격자 분석 완료: {grid_n}×{grid_n}, 점유 셀 {occupied_count}개")
    return results


def analyze_cell_occupancy(
    cell: np.ndarray,
    bg_cell: np.ndarray | None = None,
    threshold: int = 25,
    sat_threshold: int = 40,
) -> bool:
    """
    셀 이미지에서 레고 점유 여부를 판단한다.

    배경 이미지 있음 → 픽셀 차분 평균
    배경 이미지 없음 → HSV 채도 평균
    """
    if cell.size == 0:
        return False

    # 셀 중앙 60% 영역만 사용 (가장자리 그림자/빛 제외)
    h, w = cell.shape[:2]
    mh, mw = max(1, int(h * 0.2)), max(1, int(w * 0.2))
    roi = cell[mh:h - mh, mw:w - mw]

    if roi.size == 0:
        roi = cell

    if bg_cell is not None:
        bg_roi = bg_cell[mh:h - mh, mw:w - mw]
        if bg_roi.size == 0:
            bg_roi = bg_cell
        diff  = cv2.absdiff(roi, bg_roi)
        score = float(np.mean(diff))
        return score > threshold
    else:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
        return sat > sat_threshold


def estimate_cell_color(cell: np.ndarray) -> tuple[str, int, int, int]:
    """
    셀 중앙 영역의 중앙값 색상을 추정하고 LEGO 팔레트로 양자화한다.

    반환: (color_name, r, g, b)
    """
    h, w = cell.shape[:2]
    mh, mw = max(1, int(h * 0.2)), max(1, int(w * 0.2))
    roi = cell[mh:h - mh, mw:w - mw]
    if roi.size == 0:
        roi = cell

    # HSV V채널 클리핑으로 하이라이트/그림자 보정
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 40, 220)
    roi_corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    b = int(np.median(roi_corrected[:, :, 0]))
    g = int(np.median(roi_corrected[:, :, 1]))
    r = int(np.median(roi_corrected[:, :, 2]))

    snapped_bgr = snap_to_lego_color((b, g, r))
    sb, sg, sr  = snapped_bgr

    palette_bgr = np.array(list(LEGO_PALETTE.values()), dtype=np.float32)
    idx         = int(np.argmin(np.linalg.norm(palette_bgr - np.array([sb, sg, sr], dtype=np.float32), axis=1)))
    color_name  = list(LEGO_PALETTE.keys())[idx]

    return color_name, int(sr), int(sg), int(sb)
