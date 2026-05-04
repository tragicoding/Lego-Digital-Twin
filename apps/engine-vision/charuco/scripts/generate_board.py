"""
Print ChArUco board image for physical setup.

  conda run -n charuco python charuco/scripts/generate_board.py --out data/charuco_boards/board.png
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from charuco.calibration.charuco_board import generate_board_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/charuco_boards/board.png")
    parser.add_argument("--px",  type=int, default=120, help="pixels per square")
    args = parser.parse_args()
    generate_board_image(args.out, px_per_square=args.px)
