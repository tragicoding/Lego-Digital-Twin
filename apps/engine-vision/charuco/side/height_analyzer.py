"""
SIDE 뷰 워핑 이미지 → 열(column)별 높이맵 추출 모듈.

워핑된 사이드 이미지에서:
1. 실루엣 마스크 추출 (레고 있는 픽셀 = 255)
2. 각 열(x 방향)에서 최상단 점유 픽셀 찾기 → z 높이 인덱스
3. grid_n 단위로 정규화

FRONT/BACK → x-z 평면 height_map (shape: grid_n)
LEFT/RIGHT  → y-z 평면 height_map (shape: grid_n)
"""

import cv2
import numpy as np

from ..reconstruction.silhouette import extract_by_background_subtraction, _clean_mask


def extract_side_silhouette(
    warped: np.ndarray,
    bg_warped: np.ndarray | None = None,
    threshold: int = 30,
) -> np.ndarray:
    """
    워핑된 사이드뷰에서 레고 실루엣 마스크를 추출한다.

    배경 이미지 있음 → 차분 방식 (정확)
    배경 이미지 없음 → HSV 채도 기반 (폴백)

    반환: uint8 이진 마스크 (255=레고, 0=배경)
    """
    if bg_warped is not None:
        mask = extract_by_background_subtraction(warped, bg_warped, threshold)
    else:
        hsv  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sat  = hsv[:, :, 1]
        _, mask = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)
        mask = _clean_mask(mask)
    return mask


def generate_height_map(
    silhouette_mask: np.ndarray,
    grid_n: int,
) -> np.ndarray:
    """
    실루엣 마스크에서 열(column)별 최고 점유 픽셀을 찾아
    grid_n 단위 높이 인덱스 배열로 반환한다.

    반환: shape (grid_n,) int 배열
      - 값 = 0 ~ grid_n  (0 = 점유 없음)
      - 이미지 상단 픽셀 = 높은 z값 (위쪽이 높으므로 y 좌표 반전)
    """
    h, w = silhouette_mask.shape[:2]
    col_w = w / grid_n
    height_map = np.zeros(grid_n, dtype=np.int32)

    for xi in range(grid_n):
        x0 = int(xi * col_w)
        x1 = int((xi + 1) * col_w)
        col = silhouette_mask[:, x0:x1]

        # 열에서 점유된 픽셀 행 인덱스 (이미지 좌표계: 위=0)
        occupied_rows = np.where(col.max(axis=1) > 0)[0]
        if len(occupied_rows) == 0:
            height_map[xi] = 0
        else:
            topmost_row = occupied_rows.min()
            # 이미지 y → z 변환: 상단(y=0)이 z 최대
            z_ratio = 1.0 - (topmost_row / h)
            height_map[xi] = max(1, int(round(z_ratio * grid_n)))

    return height_map
