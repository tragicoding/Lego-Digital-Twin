using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using LegoTwin.Core;
using LegoTwin.Managers;

namespace LegoTwin.UI
{
    /// <summary>
    /// 종료 버튼 + 확인 다이얼로그.
    ///
    /// ※ 이 스크립트는 Button 위가 아니라 Canvas(QuitUI 루트)에 붙인다.
    ///   VR 모드에서 QuitButton을 숨겨도 Update()가 계속 동작해야 하기 때문.
    ///
    /// 씬 Hierarchy 구조:
    ///   QuitUI (Canvas) ← 이 스크립트를 붙인다
    ///     ├── QuitButton   (Button — _quitButton 연결)
    ///     └── ConfirmPanel (비활성으로 시작)
    ///          └── Panel
    ///               ├── MessageText  (TMP — "종료하시겠습니까?")
    ///               ├── YesButton    (Button)
    ///               └── NoButton     (Button)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] _quitButton   → QuitButton 연결
    ///   [ ] _confirmPanel → ConfirmPanel 연결 (비활성으로 시작)
    ///   [ ] _yesButton    → YesButton 연결
    ///   [ ] _noButton     → NoButton 연결
    ///   [ ] _vrXButtonToggle: 켜면 VR에서 왼손 X 버튼으로 패널 토글, 화면 버튼 숨김
    ///                         끄면 VR에서도 화면 버튼 그대로 표시
    /// </summary>
    public class QuitButton : MonoBehaviour
    {
        [Header("UI 참조")]
        [Tooltip("화면의 종료 버튼")]
        [SerializeField] private Button     _quitButton;
        [SerializeField] private GameObject _confirmPanel;
        [SerializeField] private Button     _yesButton;
        [SerializeField] private Button     _noButton;

        [Header("VR 옵션")]
        [Tooltip("VR에서 왼손 컨트롤러 X 버튼으로 확인 패널 토글(데스크톱은 영향 없음).")]
        [SerializeField] private bool _vrXButtonToggle = true;

        private bool _xPressedLast;

        private void Awake()
        {
            _quitButton?.onClick.AddListener(OnQuitPressed);
            _yesButton?.onClick.AddListener(OnConfirmYes);
            _noButton?.onClick.AddListener(OnConfirmNo);

            if (_confirmPanel != null) _confirmPanel.SetActive(false);
        }

        private void OnEnable()
        {
            _xPressedLast = ReadVrToggleButton();
            ApplyQuitButtonVisibility();
        }

        private void OnDestroy()
        {
            _quitButton?.onClick.RemoveListener(OnQuitPressed);
            _yesButton?.onClick.RemoveListener(OnConfirmYes);
            _noButton?.onClick.RemoveListener(OnConfirmNo);
        }

        // VR 모드이면 무조건 화면 버튼을 숨긴다(X 버튼으로만 조작). 데스크톱은 항상 표시.
        private void ApplyQuitButtonVisibility()
        {
            if (_quitButton == null) return;
            _quitButton.gameObject.SetActive(XRModeManager.Mode != AppMode.VR);
        }

        // ════════════════════════════════════════════════════════════
        // VR 컨트롤러 X 버튼 토글 (왼손 primaryButton) — VR 모드 전용
        // ════════════════════════════════════════════════════════════

        private void Update()
        {
            if (!_vrXButtonToggle) return;
            if (XRModeManager.Mode != AppMode.VR) return;

            bool pressed = ReadVrToggleButton();
            if (pressed && !_xPressedLast) TogglePanel();
            _xPressedLast = pressed;
        }

        // 왼손 컨트롤러 X 버튼(= primaryButton). 기기 미연결/미지원이면 false.
        private static bool ReadVrToggleButton()
        {
            var device = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
            return device.isValid
                && device.TryGetFeatureValue(CommonUsages.primaryButton, out bool v)
                && v;
        }

        // ── 패널 토글 / 버튼 핸들러 ─────────────────────────────────

        private void TogglePanel()
        {
            if (_confirmPanel == null) return;
            _confirmPanel.SetActive(!_confirmPanel.activeSelf);
        }

        private void OnQuitPressed()
        {
            if (_confirmPanel != null)
                _confirmPanel.SetActive(true);
            else
                Quit();
        }

        private void OnConfirmYes()
        {
            if (_confirmPanel != null) _confirmPanel.SetActive(false);

            // 현재 세션 마무리 → 씬 리로드 → 대기화면 복귀 (다음 관람객 대기)
            if (SessionManager.Instance != null)
                SessionManager.Instance.EndCurrentSession();
            else
                Quit();   // SessionManager 없을 때만 앱 종료(폴백)
        }

        private void OnConfirmNo()
        {
            if (_confirmPanel != null) _confirmPanel.SetActive(false);
        }

        private static void Quit()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
}
