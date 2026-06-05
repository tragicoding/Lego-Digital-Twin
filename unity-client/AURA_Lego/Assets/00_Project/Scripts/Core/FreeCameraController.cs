using UnityEngine;
using UnityEngine.InputSystem;   // 키보드는 IME 영향 없는 새 Input System으로 읽음
using LegoTwin.Managers;

namespace LegoTwin.Core
{
    /// <summary>
    /// 비VR(에디터/디버그·관전)용 자유 이동 카메라.
    ///
    /// XR Origin 루트의 CharacterController 를 직접 구동해
    /// 벽 충돌 · 계단 오르기(stepOffset) · 경사 · 중력 · 점프를
    /// 엔진 기본 기능으로 처리한다. (레이캐스트 지형 추적 방식 폐기)
    ///
    /// 조작 방법:
    ///   마우스 오른쪽 버튼 + 드래그 — 시선 회전 (Yaw=리그 회전, Pitch=카메라 회전)
    ///   WASD                        — 수평 이동 (벽/계단 자동 처리)
    ///   Space                       — 점프
    ///   Shift                       — 이동 속도 증가
    ///
    /// 동작 원리:
    ///   - 이 컴포넌트가 활성화되면 같은 리그의 TrackedPoseDriver(HMD 트래킹)와
    ///     XR 이동/중력 Provider 를 자동으로 비활성화한다. (카메라 직접 제어 충돌 방지)
    ///   - FreeCameraController 가 붙은 GameObject(Main Camera)의 부모에서
    ///     CharacterController 를 찾아 그 루트(XR Origin)를 이동시킨다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Main Camera (XR Origin 안)에 이 컴포넌트 추가
    ///   [ ] 부모 계층에 CharacterController 필요 (XR Origin 루트에 기본 존재)
    ///   [ ] 바닥/지형/계단에 Collider 필요
    ///   [ ] VR 빌드 배포 시 이 컴포넌트 비활성화 또는 제거 (XR 이동 복원)
    /// </summary>
    public class FreeCameraController : MonoBehaviour
    {
        [Header("이동 설정")]
        public float moveSpeed     = 10f;
        public float fastMoveSpeed = 25f;

        [Header("회전 설정")]
        public float mouseSensitivity = 2f;

        [Header("점프 / 중력 설정")]
        public float jumpHeight = 2.5f;   // 점프 최고 높이 (월드 단위)
        public float gravity    = 25f;    // 중력 가속도

        private CharacterController _cc;
        private Transform _rig;            // CharacterController 가 붙은 루트(XR Origin)
        private float _yaw;
        private float _pitch;
        private float _verticalVelocity;

        // 비VR 자유 카메라와 충돌하는 XR 컴포넌트 (타입 이름 부분 일치로 비활성화)
        private static readonly string[] ConflictingTypes =
        {
            "TrackedPoseDriver",
            "ContinuousMoveProvider", "DynamicMoveProvider", "GrabMoveProvider",
            "ContinuousTurnProvider", "SnapTurnProvider",
            "GravityProvider", "CharacterControllerDriver",
        };

        private void Awake()
        {
            _cc  = GetComponentInParent<CharacterController>();
            _rig = _cc != null ? _cc.transform : transform;

            if (_cc == null)
                Debug.LogWarning("[FreeCameraController] 부모에서 CharacterController를 찾지 못했습니다. " +
                                 "충돌/중력 없이 단순 이동으로 동작합니다.");

            DisableConflictingXrComponents();
        }

        private void Start()
        {
            _yaw   = _rig.eulerAngles.y;
            _pitch = transform.localEulerAngles.x;
            if (_pitch > 180f) _pitch -= 360f;   // 0~360 → -180~180 보정
        }

        private void Update()
        {
            HandleLook();

            // 가이드 모드 중에는 이동 잠금 (시선 회전만 허용)
            if (GameFlowManager.IsGuideMode) return;

            HandleMove();
        }

        // ── 시선 회전 (오른쪽 버튼 드래그) ─────────────────────────
        private void HandleLook()
        {
            if (Input.GetMouseButton(1))
            {
                _yaw   += Input.GetAxis("Mouse X") * mouseSensitivity;
                _pitch -= Input.GetAxis("Mouse Y") * mouseSensitivity;
                _pitch  = Mathf.Clamp(_pitch, -89f, 89f);

                _rig.rotation           = Quaternion.Euler(0f, _yaw, 0f);   // Yaw: 리그 전체
                transform.localRotation = Quaternion.Euler(_pitch, 0f, 0f); // Pitch: 카메라만

                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible   = false;
            }
            else
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible   = true;
            }
        }

        // ── 이동 · 점프 (CharacterController 구동) ───────────────────
        private void HandleMove()
        {
            float speed = (HeldKey(Key.LeftShift, KeyCode.LeftShift) || HeldKey(Key.RightShift, KeyCode.RightShift))
                ? fastMoveSpeed : moveSpeed;

            // Yaw 기준 수평 입력 방향
            Vector3 input = Vector3.zero;
            if (HeldKey(Key.W, KeyCode.W)) input += Vector3.forward;
            if (HeldKey(Key.S, KeyCode.S)) input += Vector3.back;
            if (HeldKey(Key.A, KeyCode.A)) input += Vector3.left;
            if (HeldKey(Key.D, KeyCode.D)) input += Vector3.right;

            Vector3 horizontal = Quaternion.Euler(0f, _yaw, 0f) * input.normalized * speed;

            if (_cc == null)
            {
                // 폴백: CharacterController 없으면 단순 이동 (충돌/중력 없음)
                _rig.position += horizontal * Time.deltaTime;
                return;
            }

            // 중력 + 점프
            if (_cc.isGrounded)
            {
                if (_verticalVelocity < 0f) _verticalVelocity = -2f;  // 바닥에 밀착 (isGrounded 유지)
                if (PressedKey(Key.Space, KeyCode.Space))
                    _verticalVelocity = Mathf.Sqrt(2f * jumpHeight * gravity);  // 목표 높이만큼 점프
            }
            else
            {
                _verticalVelocity -= gravity * Time.deltaTime;
            }

            // 수평 + 수직을 한 번에 Move → 벽/계단/경사/중력 자동 처리
            Vector3 velocity = horizontal + Vector3.up * _verticalVelocity;
            _cc.Move(velocity * Time.deltaTime);
        }

        // ── 키 입력 (IME 무관) ──────────────────────────────────────
        // 새 Input System(Keyboard.current)은 물리 키를 직접 읽어 한글/IME 모드에서도 동작.
        // 키보드 장치가 없으면 구 Input 으로 폴백.
        private static bool HeldKey(Key key, KeyCode fallback)
        {
            var kb = Keyboard.current;
            return kb != null ? kb[key].isPressed : Input.GetKey(fallback);
        }

        private static bool PressedKey(Key key, KeyCode fallback)
        {
            var kb = Keyboard.current;
            return kb != null ? kb[key].wasPressedThisFrame : Input.GetKeyDown(fallback);
        }

        // 비VR 자유 카메라와 충돌하는 XR 컴포넌트를 비활성화
        // (이 컴포넌트가 켜진 동안만 — VR 빌드에서는 이 스크립트를 끄면 원복)
        private void DisableConflictingXrComponents()
        {
            if (_rig == null) return;

            foreach (var mb in _rig.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (mb == null || mb == this) continue;

                string typeName = mb.GetType().Name;
                for (int i = 0; i < ConflictingTypes.Length; i++)
                {
                    if (typeName.Contains(ConflictingTypes[i]))
                    {
                        mb.enabled = false;
                        break;
                    }
                }
            }
        }
    }
}
