using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;   // 키보드는 IME 영향 없는 새 Input System으로 읽음
using UnityEngine.XR;
using LegoTwin.Core;

namespace LegoTwin.World
{
    /// <summary>
    /// 월드 공간 버튼 — 플레이어가 트리거 범위에 들어와 키를 누르면 onPress 이벤트를 발행한다.
    ///
    /// 설계 원칙
    ///   - 입력 / 동작 분리 : 이 컴포넌트는 "눌렸다"만 알린다. 무슨 일이 일어날지는
    ///                        onPress(UnityEvent)에 연결 → 열기구 탑승/하차뿐 아니라
    ///                        무엇이든 붙일 수 있다(확장성·저결합).
    ///   - IME 무관 입력    : 키를 새 Input System(Keyboard.current)으로 읽어 한글 입력
    ///                        모드에서도 정상 동작(FreeCameraController와 동일 방식).
    ///
    /// 유니티 개발자 체크리스트
    ///   [ ] 이 오브젝트에 Collider(Is Trigger ✔) — 플레이어 접근 감지 범위
    ///   [ ] _onPress 에 동작 연결 (예: BalloonRide.Enter / Exit / Toggle)
    ///   [ ] (선택) _hint 에 "E 키로 탑승" 안내 UI 연결 — 범위 안에서만 표시
    ///
    /// 비고: 플레이어 리그(CharacterController)는 매 프레임 Move(중력)로 갱신되므로
    ///       텔레포트로 범위에 들어와도 다음 프레임에 OnTriggerEnter가 정상 발생한다.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public class WorldButton : MonoBehaviour
    {
        [Header("감지")]
        [Tooltip("이 태그를 가진 오브젝트만 버튼을 누를 수 있다")]
        [SerializeField] private string _playerTag = "Player";

        [Tooltip("범위 안에서 이 키를 누르면 발동 (물리 키 — IME 무관)")]
        [SerializeField] private Key _pressKey = Key.E;

        [Header("안내(선택)")]
        [Tooltip("범위 안에 있을 때만 켜지는 안내 UI (없어도 됨)")]
        [SerializeField] private GameObject _hint;

        [Header("동작")]
        [Tooltip("키를 눌렀을 때 실행할 동작 — 여기에 BalloonRide.Enter/Exit 등을 연결")]
        [SerializeField] private UnityEvent _onPress;

        private bool _inRange;
        private bool _bPressedLast;   // VR B 버튼 edge 감지

        // 인스펙터에서 컴포넌트 추가 시 Collider를 자동으로 트리거로 설정
        private void Reset()
        {
            GetComponent<Collider>().isTrigger = true;
        }

        private void OnDisable()
        {
            _inRange = false;
            if (_hint != null) _hint.SetActive(false);
        }

        private void Update()
        {
            if (!_inRange) return;

            var kb = Keyboard.current;
            if (kb != null && kb[_pressKey].wasPressedThisFrame)
                _onPress?.Invoke();

            // VR 모드: 오른손 컨트롤러 B 버튼(secondaryButton)으로도 발동
            bool bNow = XRModeManager.Mode == AppMode.VR && ReadVrBButton();
            if (bNow && !_bPressedLast) _onPress?.Invoke();
            _bPressedLast = bNow;
        }

        private static bool ReadVrBButton()
        {
            var device = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
            return device.isValid
                && device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.secondaryButton, out bool v)
                && v;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (!other.CompareTag(_playerTag)) return;
            _inRange = true;
            _bPressedLast = ReadVrBButton();   // 진입 직전 상태 동기화 — 즉시 발동 방지
            if (_hint != null) _hint.SetActive(true);
        }

        private void OnTriggerExit(Collider other)
        {
            if (!other.CompareTag(_playerTag)) return;
            _inRange = false;
            if (_hint != null) _hint.SetActive(false);
        }

        /// <summary>
        /// 텔레포트 등 물리 트리거가 발생하지 않는 이동 후 강제로 힌트를 숨긴다.
        /// BalloonRide.OnBoarded / OnLeft 이벤트에 Inspector에서 연결한다.
        /// </summary>
        public void ForceHide()
        {
            _inRange = false;
            if (_hint != null) _hint.SetActive(false);
        }

        /// <summary>
        /// 씬의 모든 WorldButton 힌트를 강제로 숨긴다.
        /// PlayerTeleporter 등 OnTriggerExit 를 발생시키지 않는 이동 직후 호출한다.
        /// </summary>
        public static void ForceHideAll()
        {
            foreach (var wb in FindObjectsByType<WorldButton>(FindObjectsSortMode.None))
                wb.ForceHide();
        }
    }
}
