"""
TOP / SIDE 프레임 보드 생성 스크립트.

example.png 처럼 가운데가 뚫린 ChArUco 프레임 보드를 PNG로 생성한다.

원리
----
  1. 전체 크기 ChArUco 보드를 생성한다.
  2. 중앙 영역을 흰색으로 마스킹 → 프레임 모양
  3. TOP: 4면 모두 마스킹  →  사각 프레임
  4. SIDE: 아래만 열고 나머지 마스킹  →  ㄷ자형 프레임

실행 (charuco 디렉토리에서):
  # TOP 프레임 보드
  conda run -n charuco python scripts/generate_frame_board.py --type top \\
      --out data/boards/top_frame.png

  # SIDE 프레임 보드 (FRONT/BACK/LEFT/RIGHT 동일한 모양)
  conda run -n charuco python scripts/generate_frame_board.py --type side \\
      --out data/boards/side_frame.png

파라미터 (기본값):
  --squares  : 보드 전체 격자 수 (기본 10 → 10×10)
  --sq_size  : 한 칸 크기 mm (기본 30)
  --frame_w  : 프레임 두께 격자 수 (기본 2 → 2칸 = 60mm)
  --px_per_mm: 해상도 (기본 10px/mm → 30mm칸 = 300px)

출력 예상 크기 (기본값):
  보드 전체: 10×30mm = 300mm × 300mm
  가운데 뚫린 영역: (10-4)×30mm = 180mm × 180mm (LEGO BUILD AREA)
  → A4(210×297mm)에 인쇄 가능
"""

import argparse
import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from charuco.calibration.charuco_board import ARUCO_DICT, SQUARE_LEN, MARKER_LEN


def build_charuco_board(squares: int) -> aruco.CharucoBoard:
    """N×N ChArUco 보드를 생성한다."""
    return aruco.CharucoBoard(
        (squares, squares), SQUARE_LEN, MARKER_LEN, ARUCO_DICT
    )


def generate_frame_board(
    board_type: str  = "top",
    squares:    int  = 10,
    frame_w:    int  = 2,
    px_per_mm:  int  = 10,
) -> np.ndarray:
    """
    프레임 보드 이미지를 생성한다.

    Parameters
    ----------
    board_type : "top" (사각 프레임) | "side" (ㄷ자형 프레임)
    squares    : 전체 격자 수 (N×N)
    frame_w    : 프레임 두께 (격자 수)
    px_per_mm  : 해상도 (픽셀/mm)

    반환: BGR 이미지
    """
    sq_mm  = int(SQUARE_LEN * 1000)   # mm 단위 칸 크기 (기본 30mm)
    sq_px  = sq_mm * px_per_mm        # 픽셀 단위 칸 크기
    total_px = squares * sq_px

    # 전체 ChArUco 보드 이미지 생성
    board = build_charuco_board(squares)
    board_img = board.generateImage(
        (total_px, total_px), marginSize=0, borderBits=1
    )
    # 컬러 캔버스로 변환
    canvas = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    # 마스킹: 프레임 안쪽 영역을 흰색으로 채움
    inner_start = frame_w * sq_px
    inner_end   = (squares - frame_w) * sq_px

    if board_type == "top":
        # 사각 프레임: 중앙 전체 마스킹
        canvas[inner_start:inner_end, inner_start:inner_end] = 255

    elif board_type == "side":
        # ㄷ자형: 중앙 마스킹 + 하단 영역도 마스킹 (바닥 열림)
        canvas[inner_start:total_px, inner_start:inner_end] = 255

    # LEGO BUILD AREA 텍스트
    _draw_build_area_label(canvas, board_type, inner_start, inner_end, total_px)

    # 외곽 테두리
    cv2.rectangle(canvas, (0, 0), (total_px - 1, total_px - 1), (0, 0, 0), 3)

    return canvas


def _draw_build_area_label(
    canvas: np.ndarray,
    board_type: str,
    inner_start: int,
    inner_end: int,
    total_px: int,
) -> None:
    """LEGO BUILD AREA 레이블과 안내 텍스트를 그린다."""
    if board_type == "top":
        cx = (inner_start + inner_end) // 2
        cy = (inner_start + inner_end) // 2
    else:  # side
        cx = (inner_start + inner_end) // 2
        cy = (inner_start + total_px) // 2

    font      = cv2.FONT_HERSHEY_SIMPLEX
    area_w    = inner_end - inner_start

    texts = [
        ("LEGO BUILD AREA", 0.9, (30, 130, 30), 2),
    ]
    if board_type == "side":
        texts.append(("z=0 baseline (open bottom)", 0.7, (180, 50, 50), 2))

    for i, (text, scale, color, thickness) in enumerate(texts):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = cx - tw // 2
        ty = cy - (len(texts) - 1) * (th + 10) // 2 + i * (th + 15)
        cv2.putText(canvas, text, (tx, ty), font, scale, color, thickness)

    # 바닥선 (side만)
    if board_type == "side":
        cv2.line(canvas,
                 (inner_start, total_px - 3),
                 (inner_end,   total_px - 3),
                 (180, 50, 50), 4)
        cv2.putText(canvas, "<-- z=0 baseline (open, no marker here)",
                    (inner_start + 10, total_px - 10),
                    font, 0.55, (180, 50, 50), 1)


def print_marker_info(squares: int, frame_w: int) -> None:
    """생성된 보드의 마커 ID 범위와 코너 위치를 출력한다."""
    board = build_charuco_board(squares)
    # CharucoBoard marker 총 수
    total_markers = board.getChessboardSize()[0] * board.getChessboardSize()[1] // 2
    print(f"\n[정보] {squares}×{squares} 보드, 총 마커 수: {total_markers}개")
    print(f"  마커 ID 범위: 0 ~ {total_markers - 1}")
    print(f"  프레임 두께: {frame_w}칸 ({frame_w * int(SQUARE_LEN*1000)}mm)")
    inner = squares - frame_w * 2
    print(f"  LEGO BUILD AREA: {inner}×{inner}칸 ({inner * int(SQUARE_LEN*1000)}mm × {inner * int(SQUARE_LEN*1000)}mm)")
    print(f"\n  [setup.json 설정 안내]")
    print(f"  corner_ids: 네 모서리 마커 ID는 생성된 이미지에서 확인 후 기입")
    print(f"  → verify_pose.py 로 찍어보면 ID가 표시됨")


def main():
    parser = argparse.ArgumentParser(description="TOP/SIDE ChArUco 프레임 보드 생성")
    parser.add_argument("--type",       default="top",  choices=["top", "side"],
                        help="top=사각 프레임, side=ㄷ자형 프레임")
    parser.add_argument("--out",        default=None,   help="저장 경로 (미지정 시 자동)")
    parser.add_argument("--squares",    type=int, default=10, help="전체 격자 수 (NxN)")
    parser.add_argument("--frame_w",    type=int, default=2,  help="프레임 두께 (격자 수)")
    parser.add_argument("--px_per_mm",  type=int, default=10, help="해상도 px/mm")
    args = parser.parse_args()

    os.makedirs("data/boards", exist_ok=True)
    if args.out is None:
        args.out = f"data/boards/{args.type}_frame_board.png"

    print(f"[generate_frame_board] 생성 중: type={args.type}, "
          f"{args.squares}×{args.squares} 격자, 프레임 두께={args.frame_w}칸")

    img = generate_frame_board(
        board_type = args.type,
        squares    = args.squares,
        frame_w    = args.frame_w,
        px_per_mm  = args.px_per_mm,
    )

    cv2.imwrite(args.out, img)
    sq_mm   = int(SQUARE_LEN * 1000)
    total_mm = args.squares * sq_mm
    print(f"[generate_frame_board] 저장 완료 → {args.out}")
    print(f"  이미지 크기: {img.shape[1]}×{img.shape[0]}px")
    print(f"  물리 크기:   {total_mm}×{total_mm}mm  ({total_mm/10:.1f}×{total_mm/10:.1f}cm)")
    print(f"  권장 인쇄:   실제 크기(100%) / 자 대고 {sq_mm}mm 칸 크기 확인 필수")

    print_marker_info(args.squares, args.frame_w)


if __name__ == "__main__":
    main()
