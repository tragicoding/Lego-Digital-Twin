import cv2
from pathlib import Path
from datetime import datetime

# 테스트할 카메라 인덱스 범위
CAMERA_INDICES = range(0, 10)

OUTPUT_DIR = Path("camera_test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def test_camera(index: int):
    print(f"\n[Camera {index}] opening...")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[Camera {index}] NOT AVAILABLE")
        return

    # 해상도는 필요하면 조정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[Camera {index}] OPENED")
    print("  Press SPACE to save image")
    print("  Press N to next camera")
    print("  Press Q to quit")

    window_name = f"Camera Index {index}"

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            print(f"[Camera {index}] failed to read frame")
            break

        label = f"INDEX {index}"
        cv2.putText(
            frame,
            label,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"camera_{index}_{timestamp}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[Camera {index}] saved: {path}")

        elif key == ord("n"):
            print(f"[Camera {index}] next")
            break

        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt

    cap.release()
    cv2.destroyWindow(window_name)


def main():
    print("USB Camera Index Tester")
    print("=======================")
    print("각 카메라 화면을 보면서 어떤 위치인지 기록하세요.")
    print("예: index 0 = front, index 1 = left ...")

    try:
        for index in CAMERA_INDICES:
            test_camera(index)
    except KeyboardInterrupt:
        print("\nquit")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()