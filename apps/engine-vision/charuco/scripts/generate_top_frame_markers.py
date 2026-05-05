"""
TOP 뷰 사각형 프레임용 ArUco 마커 생성.

마커 배치 (위에서 본 모습):
  ┌──[0]────────[1]──┐
  │                  │
  [7]  LEGO BUILD  [2]
  │      AREA       │
  [6]               [3]
  │                  │
  └──[5]────────[4]──┘

마커 ID 할당:
  0: 좌상단 코너   1: 우상단 코너
  2: 우측 중간     3: 우하단 코너
  4: 하단 우측     5: 하단 중간
  6: 좌측 중간     7: 좌하단 코너

  → 코너 4개 (0,1,3,5) 필수 / 중간 4개 (2,4,6,7) 선택 (정밀도 향상)

실행 (charuco 디렉토리에서):
  conda run -n charuco python scripts/generate_top_frame_markers.py \\
      --out data/markers/top \\
      --size 150

출력:
  data/markers/top/marker_00.png ~ marker_07.png  (개별 마커, 잘라서 배치)
  data/markers/top/layout_guide.png               (배치 가이드 이미지)
"""

import argparse
import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from charuco.calibration.charuco_board import ARUCO_DICT

# TOP 프레임 마커 ID 및 배치 설명
TOP_MARKERS = {
    0: "좌상단 코너",
    1: "우상단 코너",
    2: "우측 중간",
    3: "우하단 코너",
    4: "하단 우측",
    5: "하단 중간",
    6: "좌측 중간",
    7: "좌하단 코너",
}

# 필수 코너 4개 — setup.json corner_ids 와 반드시 일치
CORNER_IDS = [0, 1, 3, 5]  # [좌상단, 우상단, 우하단, 좌하단]


def generate_marker(marker_id: int, size_px: int) -> np.ndarray:
    """단일 ArUco 마커 이미지를 생성한다."""
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    aruco.generateImageMarker(ARUCO_DICT, marker_id, size_px, img, 1)
    return img


def add_label(img_gray: np.ndarray, marker_id: int, label: str) -> np.ndarray:
    """마커 이미지 아래에 ID와 설명 텍스트를 추가한다."""
    h, w = img_gray.shape
    canvas = np.ones((h + 60, w), dtype=np.uint8) * 255
    canvas[:h, :] = img_gray
    cv2.putText(canvas, f"ID: {marker_id:02d}", (5, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 2)
    cv2.putText(canvas, label, (5, h + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 80, 1)
    return canvas


def generate_layout_guide(size_px: int) -> np.ndarray:
    """TOP 프레임 마커 배치 가이드 이미지를 생성한다."""
    canvas_px = 700
    canvas    = np.ones((canvas_px, canvas_px, 3), dtype=np.uint8) * 245

    margin   = 60
    frame_w  = canvas_px - margin * 2
    build_gap = frame_w // 5

    # 외곽 프레임
    cv2.rectangle(canvas,
                  (margin, margin),
                  (margin + frame_w, margin + frame_w),
                  (180, 180, 180), 2)

    # LEGO BUILD AREA
    bx0, by0 = margin + build_gap, margin + build_gap
    bx1, by1 = margin + frame_w - build_gap, margin + frame_w - build_gap
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (80, 180, 80), 2)
    cv2.putText(canvas, "LEGO BUILD AREA",
                (bx0 + 10, by0 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 130, 40), 2)

    fw = frame_w
    cx, cy = margin + fw // 2, margin + fw // 2

    # 마커 위치 (프레임 경계 위)
    positions = {
        0: (margin,          margin),
        1: (margin + fw,     margin),
        2: (margin + fw,     cy),
        3: (margin + fw,     margin + fw),
        4: (cx + build_gap,  margin + fw),
        5: (cx,              margin + fw),
        6: (margin,          cy),
        7: (margin,          margin + fw),
    }

    thumb_size = 55
    for mid, (px, py) in positions.items():
        marker_img = generate_marker(mid, size_px)
        thumb      = cv2.resize(marker_img, (thumb_size, thumb_size))
        x0 = max(0, px - thumb_size // 2)
        y0 = max(0, py - thumb_size // 2)
        x1 = min(canvas_px, x0 + thumb_size)
        y1 = min(canvas_px, y0 + thumb_size)
        region = cv2.cvtColor(thumb[:y1 - y0, :x1 - x0], cv2.COLOR_GRAY2BGR)
        canvas[y0:y1, x0:x1] = region

        color = (30, 30, 200) if mid in CORNER_IDS else (180, 80, 20)
        cv2.putText(canvas, f"[{mid}]",
                    (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 범례
    cv2.putText(canvas, "파란색=필수 코너(ID 0,1,3,5)  주황색=선택 중간(ID 2,4,6,7)",
                (10, canvas_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="TOP 프레임 ArUco 마커 생성")
    parser.add_argument("--out",  default="data/markers/top")
    parser.add_argument("--size", type=int, default=150, help="마커 크기 px")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[top_frame] 마커 생성 (크기: {args.size}px)")
    for mid, label in TOP_MARKERS.items():
        img      = generate_marker(mid, args.size)
        labeled  = add_label(img, mid, label)
        path     = os.path.join(args.out, f"marker_{mid:02d}.png")
        cv2.imwrite(path, labeled)
        tag = " ← 필수" if mid in CORNER_IDS else ""
        print(f"  ID {mid:02d} ({label}){tag} → {path}")

    guide = generate_layout_guide(args.size)
    guide_path = os.path.join(args.out, "layout_guide.png")
    cv2.imwrite(guide_path, guide)

    print(f"\n[top_frame] 배치 가이드 → {guide_path}")
    print("\n배치 방법:")
    print("  1. marker_XX.png 각각 잘라서 두꺼운 종이/라미네이팅")
    print("  2. layout_guide.png 보면서 프레임 해당 위치에 고정")
    print("  3. ID 0,1,3,5 코너 4개는 필수 / 나머지는 선택")
    print(f"\n  setup.json corner_ids 값: {CORNER_IDS}  (변경 금지)")


if __name__ == "__main__":
    main()
