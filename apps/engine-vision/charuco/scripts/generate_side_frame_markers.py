"""
SIDE 뷰 ㄷ자형 프레임용 ArUco 마커 생성.

마커 배치 (정면에서 본 모습):
  ┌──[top_L]────[top_R]──┐
  │                      │
  [side_L]  레고 영역  [side_R]
  │                      │
  └                      ┘  ← 하단 없음 (z=0 기준선)

뷰별 마커 ID:
  FRONT: top_ids=[4,5]   side_ids=[6,7]
  BACK:  top_ids=[8,9]   side_ids=[10,11]
  LEFT:  top_ids=[12,13] side_ids=[14,15]
  RIGHT: top_ids=[16,17] side_ids=[18,19]

실행 (charuco 디렉토리에서):
  conda run -n charuco python scripts/generate_side_frame_markers.py \\
      --out  data/markers/side \\
      --size 150

출력:
  side/FRONT/marker_04.png ~ marker_07.png  + layout_guide.png
  side/BACK/ ...
  side/LEFT/ ...
  side/RIGHT/ ...
"""

import argparse
import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from charuco.calibration.charuco_board import ARUCO_DICT

# 뷰별 마커 ID 정의 (setup.json 과 반드시 일치)
SIDE_MARKER_IDS = {
    "FRONT": {"top_ids": [4,  5 ], "side_ids": [6,  7 ]},
    "BACK":  {"top_ids": [8,  9 ], "side_ids": [10, 11]},
    "LEFT":  {"top_ids": [12, 13], "side_ids": [14, 15]},
    "RIGHT": {"top_ids": [16, 17], "side_ids": [18, 19]},
}

# 마커 위치 레이블
POSITION_LABELS = {
    "top_ids":  ["상단 좌측", "상단 우측"],
    "side_ids": ["측면 좌측", "측면 우측"],
}


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


def generate_side_layout_guide(view: str, ids: dict, size_px: int) -> np.ndarray:
    """ㄷ자형 프레임 배치 가이드 이미지를 생성한다."""
    canvas_w, canvas_h = 700, 600
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245

    ml, mt, mr, mb = 80, 60, 80, 120   # 마진
    fw = canvas_w - ml - mr
    fh = canvas_h - mt - mb

    # 상단 바
    cv2.rectangle(canvas, (ml, mt), (ml + fw, mt + 40), (200, 200, 220), -1)
    cv2.rectangle(canvas, (ml, mt), (ml + fw, mt + 40), (120, 120, 180), 2)

    # 좌측 바
    cv2.rectangle(canvas, (ml, mt), (ml + 40, mt + fh), (200, 200, 220), -1)
    cv2.rectangle(canvas, (ml, mt), (ml + 40, mt + fh), (120, 120, 180), 2)

    # 우측 바
    cv2.rectangle(canvas, (ml + fw - 40, mt), (ml + fw, mt + fh), (200, 200, 220), -1)
    cv2.rectangle(canvas, (ml + fw - 40, mt), (ml + fw, mt + fh), (120, 120, 180), 2)

    # 레고 영역
    bx0, by0 = ml + 40, mt + 40
    bx1, by1 = ml + fw - 40, mt + fh
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (80, 180, 80), 2)
    cv2.putText(canvas, "LEGO SIDE AREA",
                (bx0 + 10, by0 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 130, 40), 2)

    # 바닥 기준선 z=0
    cv2.line(canvas, (ml, mt + fh), (ml + fw, mt + fh), (50, 50, 200), 2)
    cv2.putText(canvas, "바닥선 z=0 (setup.json z0_pixel_y 측정 위치)",
                (ml, mt + fh + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 200), 1)

    # 마커 위치 표시
    thumb_size = 55
    tl_id, tr_id = ids["top_ids"]
    ll_id, rl_id = ids["side_ids"]

    marker_positions = {
        tl_id: (ml,          mt,                "상단 좌측"),
        tr_id: (ml + fw - thumb_size, mt,        "상단 우측"),
        ll_id: (ml,          mt + fh // 2,      "측면 좌측"),
        rl_id: (ml + fw - thumb_size, mt + fh // 2, "측면 우측"),
    }

    for mid, (px, py, pos_label) in marker_positions.items():
        marker_img = generate_marker(mid, size_px)
        thumb = cv2.resize(marker_img, (thumb_size, thumb_size))
        x0 = max(0, min(canvas_w - thumb_size, px))
        y0 = max(0, min(canvas_h - thumb_size, py))
        canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = \
            cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

        color = (30, 30, 180) if mid in ids["top_ids"] else (180, 80, 20)
        cv2.putText(canvas, f"[{mid:02d}] {pos_label}",
                    (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # 제목
    cv2.putText(canvas, f"{view} VIEW — ㄷ자형 프레임 배치 가이드",
                (ml, mt - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    cv2.putText(canvas, "파란색=상단 마커  주황색=측면 마커",
                (10, canvas_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="SIDE ㄷ자형 프레임 ArUco 마커 생성")
    parser.add_argument("--out",   default="data/markers/side")
    parser.add_argument("--size",  type=int, default=150, help="마커 크기 px")
    parser.add_argument("--views", nargs="+",
                        default=["FRONT", "BACK", "LEFT", "RIGHT"],
                        help="생성할 뷰 선택")
    args = parser.parse_args()

    for view in args.views:
        if view not in SIDE_MARKER_IDS:
            print(f"  알 수 없는 뷰: {view}, 스킵")
            continue

        out_dir = os.path.join(args.out, view)
        os.makedirs(out_dir, exist_ok=True)
        ids = SIDE_MARKER_IDS[view]

        print(f"\n[{view}] 마커 생성 중...")
        for group, labels in POSITION_LABELS.items():
            for i, (mid, label) in enumerate(zip(ids[group], labels)):
                img     = generate_marker(mid, args.size)
                labeled = add_label(img, mid, label)
                path    = os.path.join(out_dir, f"marker_{mid:02d}.png")
                cv2.imwrite(path, labeled)
                print(f"  ID {mid:02d} ({label}) → {path}")

        guide      = generate_side_layout_guide(view, ids, args.size)
        guide_path = os.path.join(out_dir, "layout_guide.png")
        cv2.imwrite(guide_path, guide)
        print(f"  배치 가이드 → {guide_path}")

    print("\n배치 방법:")
    print("  1. 각 marker_XX.png 잘라서 두꺼운 종이/라미네이팅")
    print("  2. 각 뷰의 layout_guide.png 보면서 해당 위치에 고정")
    print("  3. 바닥선 z=0: 카메라로 찍었을 때 픽셀 y좌표를 setup.json z0_pixel_y에 기록")
    print("\n  setup.json side_views ID 값:", SIDE_MARKER_IDS)


if __name__ == "__main__":
    main()
