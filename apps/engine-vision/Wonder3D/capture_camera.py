import cv2
import os
from datetime import datetime
# WSL 네트워크 주소로 변경합니다.
# 주의: 윈도우 경로는 백슬래시(\)를 쓰기 때문에 문자열 앞에 반드시 'r'을 붙여야 합니다. (Raw String)
WSL_INPUT_DIR = r"\\wsl.localhost\Ubuntu\home\jskim\Lego-Digital-Twin\apps\engine-vision\Wonder3D\inputs"

def start_camera_capture(save_dir=WSL_INPUT_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lego_capture_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    # 2. 웹캠 연결 (0번은 기본 연결된 첫 번째 카메라를 의미합니다)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] 카메라를 찾을 수 없습니다. USB 연결을 확인해주세요.")
        return None

    print("="*50)
    print("  📸 Lego Digital Twin - 카메라 모듈 가동 📸")
    print("  - [Spacebar] : 사진 촬영 및 저장")
    print("  - [ESC] 또는 'q' : 종료")
    print("="*50)

    while True:
        # 3. 카메라로부터 프레임 읽어오기
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽어올 수 없습니다.")
            break

        # 화면에 카메라 영상 출력
        cv2.imshow("Lego Digital Twin - Camera Preview", frame)

        # 4. 키보드 입력 대기 (1ms 마다 체크)
        key = cv2.waitKey(1) & 0xFF

        # [Spacebar]를 누르면 (아스키코드 32)
        if key == 32:
            cv2.imwrite(save_path, frame)
            print(f"[SUCCESS] 찰칵! 사진이 성공적으로 저장되었습니다: {save_path}")
            break
            
        # [ESC] (아스키코드 27) 또는 'q'를 누르면
        elif key == 27 or key == ord('q'):
            print("[INFO] 촬영을 취소하고 종료합니다.")
            break

    # 5. 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()
    return save_path

if __name__ == "__main__":
    start_camera_capture()