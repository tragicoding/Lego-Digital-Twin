"""
Visualise ChArUco pose for each view image.
Draws coordinate axes on the image to confirm correct detection.

  conda run -n charuco python charuco/scripts/verify_pose.py \\
      --image data/capture/FRONT.jpg \\
      --params data/camera_params.npz
"""
import argparse
import sys, os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from charuco.calibration.camera_calibrator import load_params
from charuco.calibration.charuco_board     import detect_charuco, draw_axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--params", default="data/camera_params.npz")
    args = parser.parse_args()

    K, D     = load_params(args.params)
    image    = cv2.imread(args.image)
    rvec, tvec, corners, ids = detect_charuco(image, K, D)

    if rvec is None:
        print("Detection failed — check lighting and board visibility")
        cv2.imshow("Result", image)
    else:
        print(f"rvec={rvec.ravel().round(4)}  tvec={tvec.ravel().round(4)}")
        vis = draw_axis(image, rvec, tvec, K, D, length=0.05)
        cv2.imshow("Result", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
