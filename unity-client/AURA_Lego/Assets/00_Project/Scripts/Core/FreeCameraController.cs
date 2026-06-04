using UnityEngine;
using LegoTwin.Managers;

namespace LegoTwin.Core
{
    /// <summary>
    /// 씬 뷰 방식의 자유 시점 카메라 컨트롤러.
    ///
    /// 조작 방법:
    ///   마우스 오른쪽 버튼 + 드래그 — 시선 회전
    ///   마우스 오른쪽 버튼 + WASD  — 전후좌우 이동
    ///   마우스 오른쪽 버튼 + Q/E   — 위/아래 이동
    ///   Shift                       — 이동 속도 증가
    ///   마우스 스크롤               — 앞뒤 이동 (빠른 줌)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Main Camera (또는 카메라를 담은 GameObject)에 이 컴포넌트 추가
    ///   [ ] VR 빌드 배포 시 이 컴포넌트 비활성화 또는 제거
    /// </summary>
    public class FreeCameraController : MonoBehaviour
    {
        [Header("이동 설정")]
        public float moveSpeed     = 10f;
        public float fastMoveSpeed = 30f;
        public float scrollSpeed   = 20f;

        [Header("회전 설정")]
        public float mouseSensitivity = 2f;

        private float _yaw;
        private float _pitch;

        private void Start()
        {
            var euler = transform.eulerAngles;
            _yaw   = euler.y;
            _pitch = euler.x;
        }

        private void Update()
        {
            bool rightHeld = Input.GetMouseButton(1);

            // ── 시선 회전 (오른쪽 버튼 드래그) ─────────────────────────
            if (rightHeld)
            {
                _yaw   += Input.GetAxis("Mouse X") * mouseSensitivity;
                _pitch -= Input.GetAxis("Mouse Y") * mouseSensitivity;
                _pitch  = Mathf.Clamp(_pitch, -89f, 89f);

                transform.eulerAngles = new Vector3(_pitch, _yaw, 0f);

                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible   = false;
            }
            else
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible   = true;
            }

            // ── WASD 이동 · 스크롤 (가이드 모드 중 잠금) ────────────────
            if (!GameFlowManager.IsGuideMode)
            {
                if (rightHeld)
                {
                    float speed = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)
                        ? fastMoveSpeed : moveSpeed;

                    Vector3 dir = Vector3.zero;
                    if (Input.GetKey(KeyCode.W)) dir += transform.forward;
                    if (Input.GetKey(KeyCode.S)) dir -= transform.forward;
                    if (Input.GetKey(KeyCode.A)) dir -= transform.right;
                    if (Input.GetKey(KeyCode.D)) dir += transform.right;
                    if (Input.GetKey(KeyCode.E)) dir += Vector3.up;
                    if (Input.GetKey(KeyCode.Q)) dir -= Vector3.up;

                    transform.position += dir * speed * Time.deltaTime;
                }

                float scroll = Input.GetAxis("Mouse ScrollWheel");
                if (Mathf.Abs(scroll) > 0.001f)
                    transform.position += transform.forward * scroll * scrollSpeed;
            }
        }

    }
}
