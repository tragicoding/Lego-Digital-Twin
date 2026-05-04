"""
Interactive calibration image capture using webcam.

Move the ChArUco board around to different angles.
Press SPACE to save a frame, Q to quit and run calibration.

  conda run -n charuco python charuco/scripts/capture_calibration.py \\
      --out data/calib --camera 0 --n 20
"""
import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from charuco.calibration.charuco_board import ARUCO_DICT, BOARD
import cv2.aruco as aruco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="data/calib")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--n",      type=int, default=20, help="frames to collect")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap     = cv2.VideoCapture(args.camera)
    count   = 0
    charuco_det = aruco.CharucoDetector(BOARD)

    print("Show the ChArUco board at different angles. SPACE=save  Q=quit")
    while count < args.n:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = charuco_det.detectBoard(gray)
        display = frame.copy()

        if corners is not None and len(corners) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, (0, 255, 0))
            cv2.putText(display, f"Detected {len(corners)} corners", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display, f"Saved: {count}/{args.n}  SPACE=save  Q=quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and corners is not None and len(corners) >= 6:
            path = os.path.join(args.out, f"calib_{count:03d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Saved frame {count}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} frames in {args.out}")


if __name__ == "__main__":
    main()
