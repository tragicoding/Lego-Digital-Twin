using UnityEngine;
using UnityEngine.UI;
using LegoTwin.Managers;

namespace LegoTwin.UI
{
    /// <summary>
    /// 종료 버튼 + 확인 다이얼로그.
    ///
    /// 씬 Hierarchy 구조:
    ///   QuitUI (Canvas — Screen Space Overlay)
    ///     ├── QuitButton   (Button + 이 스크립트)
    ///     └── ConfirmPanel (비활성으로 시작)
    ///          └── Panel
    ///               ├── MessageText  (TMP — "종료하시겠습니까?")
    ///               ├── YesButton    (Button)
    ///               └── NoButton     (Button)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] _confirmPanel → ConfirmPanel 연결 (비활성으로 시작)
    ///   [ ] _yesButton    → YesButton 연결
    ///   [ ] _noButton     → NoButton 연결
    ///   [ ] QuitButton의 OnClick → 비워도 됨 (Awake 자동 연결)
    /// </summary>
    [RequireComponent(typeof(Button))]
    public class QuitButton : MonoBehaviour
    {
        [Header("확인 다이얼로그")]
        [SerializeField] private GameObject _confirmPanel;
        [SerializeField] private Button     _yesButton;
        [SerializeField] private Button     _noButton;

        private void Awake()
        {
            GetComponent<Button>().onClick.AddListener(OnQuitPressed);
            _yesButton?.onClick.AddListener(OnConfirmYes);
            _noButton?.onClick.AddListener(OnConfirmNo);

            if (_confirmPanel != null) _confirmPanel.SetActive(false);
        }

        private void OnDestroy()
        {
            GetComponent<Button>().onClick.RemoveListener(OnQuitPressed);
            _yesButton?.onClick.RemoveListener(OnConfirmYes);
            _noButton?.onClick.RemoveListener(OnConfirmNo);
        }

        // ── 버튼 핸들러 ──────────────────────────────────────────────

        private void OnQuitPressed()
        {
            if (_confirmPanel != null)
                _confirmPanel.SetActive(true);
            else
                Quit();  // 패널 미연결 시 바로 종료
        }

        private void OnConfirmYes()
        {
            if (_confirmPanel != null) _confirmPanel.SetActive(false);

            // 큐 전진 후 다음 세션 로드 — 다음 세션 없으면 앱 종료
            if (SessionManager.Instance != null)
                SessionManager.Instance.AdvanceAndLoadNext(onNoNext: Quit);
            else
                Quit();
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
