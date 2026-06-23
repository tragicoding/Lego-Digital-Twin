using UnityEngine;
using UnityEngine.InputSystem;   // 키보드는 IME 영향 없는 새 Input System으로 읽음
using UnityEngine.EventSystems;
using TMPro;
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
    ///   - 실행 모드는 XRModeManager(단일 소스)가 시작 시 1회 판정한다.
    ///       · 데스크톱(키마) 모드 → 이 컴포넌트가 동작하며, 같은 리그의 TrackedPoseDriver
    ///         (HMD 트래킹)·XR 이동/중력 Provider 를 비활성화한다(카메라 직접 제어 충돌 방지).
    ///       · VR 모드            → 이 컴포넌트가 스스로 꺼지고 XR 트래킹/이동을 그대로 둔다.
    ///     XRModeManager 가 없으면 _modeResolveTimeout 후 데스크톱으로 폴백(무회귀).
    ///   - FreeCameraController 가 붙은 GameObject(Main Camera)의 부모에서
    ///     CharacterController 를 찾아 그 루트(XR Origin)를 이동시킨다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Main Camera (XR Origin 안)에 이 컴포넌트 추가
    ///   [ ] 부모 계층에 CharacterController 필요 (XR Origin 루트에 기본 존재)
    ///   [ ] 바닥/지형/계단에 Collider 필요
    ///   [ ] 씬에 XRModeManager 배치 (VR↔키마 자동 판정의 단일 소스)
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

        [Header("실행 모드")]
        [Tooltip("XRModeManager 가 씬에 없을 때 이 시간(초) 후 데스크톱(키마)으로 폴백. " +
                 "매니저가 있으면(대기영상 게이트 동안 확정 보류 포함) 확정될 때까지 무한 대기한다.")]
        public float modeResolveTimeout = 4f;

        private CharacterController _cc;
        private Transform _rig;            // CharacterController 가 붙은 루트(XR Origin)
        private float _yaw;
        private float _pitch;
        private float _verticalVelocity;

        // 실행 모드(VR/데스크톱) 적용 상태 — XRModeManager 확정 전엔 카메라 조작을 보류한다.
        private bool _modeApplied;
        private AppMode _appliedMode;
        private float _modeWaitTimer;

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

            // XR 비활성화는 더 이상 Awake에서 무조건 하지 않는다.
            // 데스크톱 모드로 확정될 때만 끈다(ApplyMode). VR 모드면 XR 트래킹을 그대로 둔다.
        }

        // 현재 리그(yaw)·카메라(pitch) 회전에서 _yaw/_pitch 를 다시 읽어온다.
        // Start 초기화 + 드래그 시작 시 재동기화(텔레포트 등 외부 회전 변경 흡수)에 공통 사용.
        private void SyncLookFromTransforms()
        {
            _yaw   = _rig.eulerAngles.y;
            _pitch = transform.localEulerAngles.x;
            if (_pitch > 180f) _pitch -= 360f;   // 0~360 → -180~180 보정
        }

        private void Update()
        {
            // 모드 확정 전에는 카메라를 조작하지 않는다(잘못된 모드로 한 프레임도 움직이지 않게).
            if (!_modeApplied)
            {
                TryResolveMode();
                if (!_modeApplied) return;
            }

            // VR 모드면 XR 리그가 카메라를 구동하므로 자유 카메라는 정지한다.
            if (_appliedMode != AppMode.DesktopKBM) { enabled = false; return; }

            HandleLook();

            // 가이드 모드 중 또는 입력창 타이핑 중에는 WASD·점프를 잠근다.
            // 단, 중력/지면 안착은 항상 적용해 두 모드의 눈높이(키)가 동일하게 유지되도록 한다.
            // (새 Input System은 물리 키를 직접 읽어 UI 포커스를 무시하므로 명시적으로 차단)
            bool allowInput = !GameFlowManager.IsGuideMode && !IsTypingInInputField();

            HandleMove(allowInput);
        }

        // 현재 TMP_InputField가 포커스되어 텍스트 입력 중인지 확인
        private static bool IsTypingInInputField()
        {
            var es = EventSystem.current;
            var go = es != null ? es.currentSelectedGameObject : null;
            if (go == null) return false;

            var field = go.GetComponent<TMP_InputField>();
            return field != null && field.isFocused;
        }

        // ── 시선 회전 (오른쪽 버튼 드래그) ─────────────────────────
        private void HandleLook()
        {
            // 드래그 시작 프레임: 현재 실제 회전으로 _yaw/_pitch 재동기화.
            // 텔레포트 등 외부에서 리그를 돌렸어도 그 방향을 이어받아, 시선 시작 시 스냅(튐)을 방지한다.
            if (Input.GetMouseButtonDown(1))
                SyncLookFromTransforms();

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
        // allowInput=false면 중력/지면 안착만 적용하고 WASD·점프는 무시한다.
        private void HandleMove(bool allowInput)
        {
            // Yaw 기준 수평 입력 방향 (입력 허용 시에만)
            Vector3 horizontal = Vector3.zero;
            if (allowInput)
            {
                float speed = (HeldKey(Key.LeftShift, KeyCode.LeftShift) || HeldKey(Key.RightShift, KeyCode.RightShift))
                    ? fastMoveSpeed : moveSpeed;

                Vector3 input = Vector3.zero;
                if (HeldKey(Key.W, KeyCode.W)) input += Vector3.forward;
                if (HeldKey(Key.S, KeyCode.S)) input += Vector3.back;
                if (HeldKey(Key.A, KeyCode.A)) input += Vector3.left;
                if (HeldKey(Key.D, KeyCode.D)) input += Vector3.right;

                horizontal = Quaternion.Euler(0f, _yaw, 0f) * input.normalized * speed;
            }

            if (_cc == null)
            {
                // 폴백: CharacterController 없으면 단순 이동 (충돌/중력 없음)
                _rig.position += horizontal * Time.deltaTime;
                return;
            }

            // 중력은 항상 적용 (두 모드 눈높이 일치) / 점프는 입력 허용 시에만
            if (_cc.isGrounded)
            {
                if (_verticalVelocity < 0f) _verticalVelocity = -2f;  // 바닥에 밀착 (isGrounded 유지)
                if (allowInput && PressedKey(Key.Space, KeyCode.Space))
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

        // ── 실행 모드 해석 (XRModeManager 단일 소스) ────────────────
        // 매니저가 확정하면 그 모드를 적용. 매니저가 없거나 늦으면 데스크톱으로 폴백(무회귀).
        private void TryResolveMode()
        {
            if (XRModeManager.Resolved) { ApplyMode(XRModeManager.Mode); return; }

            // 매니저가 씬에 있으면 확정을 맡긴다(대기영상 게이트 동안엔 일부러 확정을 미룬다 →
            // 그동안 헤드셋을 쓰면 VR 로 확정됨). 스스로 데스크톱으로 확정해 그 기회를 빼앗지 않는다.
            if (XRModeManager.Instance != null) return;

            // 매니저가 아예 없을 때만 안전 폴백(기존 키마 흐름 무회귀).
            _modeWaitTimer += Time.unscaledDeltaTime;
            if (_modeWaitTimer >= modeResolveTimeout) ApplyMode(AppMode.DesktopKBM);
        }

        private void ApplyMode(AppMode mode)
        {
            _modeApplied = true;
            _appliedMode = mode;

            if (mode == AppMode.DesktopKBM)
            {
                DisableConflictingXrComponents();   // 키마 모드일 때만 XR 트래킹/이동을 끈다
                SyncLookFromTransforms();           // 이동 시작 전 시선(_yaw/_pitch) 동기화
            }
            // VR 모드: 아무것도 끄지 않는다. 다음 Update에서 enabled=false 로 자유 카메라 정지.
        }

        // 비VR 자유 카메라와 충돌하는 XR 컴포넌트를 비활성화
        // (데스크톱 모드로 확정됐을 때만 — VR 모드/이 스크립트 비활성 시 원복)
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
