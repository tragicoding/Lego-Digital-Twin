using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace LegoTwin.UI
{
    /// <summary>
    /// 모션 프롬프트 입력 UI.
    ///
    /// 역할:
    ///   - GuideNPCController.OnMotionPromptRequested 이벤트 발생 시 입력창 표시
    ///   - 플레이어가 텍스트를 입력하고 확인 버튼(또는 Enter)을 누르면 callback 호출
    ///   - callback(inputText) → MotionPromptParser → AnimationClip 재생
    ///
    /// 씬 Hierarchy 구조:
    ///   MotionPromptUI          (Canvas)
    ///     └── PopupRoot         (GameObject + CanvasGroup)
    ///          └── Panel        (Image — 배경)
    ///               ├── TitleText    (TMP_Text — "원하는 동작을 입력하세요")
    ///               ├── InputField   (TMP_InputField — 텍스트 입력)
    ///               ├── SubmitButton (Button)
    ///               │    └── ButtonText (TMP_Text — "확인")
    ///               └── ExitButton   (Button — "나가기", 선택. 이름이 "ExitButton"이면 자동 연결)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] _popupRoot   — PopupRoot GameObject 연결
    ///   [ ] _inputField  — TMP_InputField 연결
    ///   [ ] _submitButton — Button 연결
    ///   [ ] _exitButton  — (선택) 나가기 Button 연결 (미연결 시 "ExitButton" 이름으로 자동 탐색)
    ///   [ ] _canvasGroup — PopupRoot 의 CanvasGroup 연결
    ///   [ ] GameFlowManager._motionPromptUI 에 이 GameObject 드래그
    /// </summary>
    public class MotionPromptUI : MonoBehaviour
    {
        [Header("UI 요소")]
        [SerializeField] private GameObject     _popupRoot;
        [SerializeField] private TMP_Text       _titleText;
        [SerializeField] private TMP_InputField _inputField;
        [SerializeField] private Button         _submitButton;
        [Tooltip("나가기 버튼 (선택). 입력을 취소할 수 있어야 하는 자유 모드에서 표시된다.")]
        [SerializeField] private Button         _exitButton;
        [SerializeField] private CanvasGroup    _canvasGroup;

        [Header("연출 설정")]
        [SerializeField] private float _fadeDuration = 0.2f;

        private Action<string> _pendingCallback;
        private Action         _pendingExit;
        private System.Collections.IEnumerator _fadeRoutine;
        private string _defaultTitle;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Awake()
        {
            if (_titleText == null && _popupRoot != null)
            {
                var texts = _popupRoot.GetComponentsInChildren<TMP_Text>(true);
                foreach (var text in texts)
                {
                    if (text != null && text.gameObject.name == "TitleText")
                    {
                        _titleText = text;
                        break;
                    }
                }
            }

            // 나가기 버튼 미연결 시 PopupRoot 자식 중 이름이 "ExitButton"인 Button 자동 탐색
            if (_exitButton == null && _popupRoot != null)
            {
                foreach (var btn in _popupRoot.GetComponentsInChildren<Button>(true))
                {
                    if (btn != null && btn.gameObject.name == "ExitButton")
                    {
                        _exitButton = btn;
                        break;
                    }
                }
            }

            if (_popupRoot != null)
                _popupRoot.SetActive(false);

            if (_canvasGroup != null)
                _canvasGroup.alpha = 0f;

            if (_titleText != null)
                _defaultTitle = _titleText.text;

            _submitButton?.onClick.AddListener(OnSubmit);

            if (_exitButton != null)
            {
                _exitButton.onClick.AddListener(OnExit);
                _exitButton.gameObject.SetActive(false);
            }
        }

        private void Update()
        {
            // PopupRoot가 활성 상태일 때 Enter 키로도 제출 가능
            if (_popupRoot != null && _popupRoot.activeSelf)
            {
                if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
                    OnSubmit();
            }
        }

        private void OnDestroy()
        {
            _submitButton?.onClick.RemoveListener(OnSubmit);
            if (_exitButton != null) _exitButton.onClick.RemoveListener(OnExit);
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API — GameFlowManager 에서 호출
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 입력 UI를 표시하고 콜백을 등록한다.
        /// 플레이어가 확인 버튼 또는 Enter를 누르면 callback(입력 텍스트) 가 호출된다.
        /// </summary>
        public void Show(Action<string> callback)
        {
            Show(callback, null, null, false, null);
        }

        /// <summary>
        /// 입력 UI를 표시하고 제목/초기값을 선택적으로 덮어쓴다.
        /// 자유 모드에서는 같은 입력 UI를 bubble_text 편집에도 재사용한다.
        /// </summary>
        public void Show(Action<string> callback, string title, string initialValue)
        {
            Show(callback, title, initialValue, false, null);
        }

        /// <summary>
        /// 입력 UI를 표시한다. showExitButton 이 true 면 나가기 버튼을 함께 노출하고,
        /// 나가기를 누르면 입력을 제출하지 않고 onExit 을 호출한다.
        /// (SignatureMotionConfirmUI 의 나가기 버튼과 동일한 패턴)
        /// </summary>
        public void Show(Action<string> callback, string title, string initialValue,
                         bool showExitButton, Action onExit)
        {
            _pendingCallback = callback;
            _pendingExit     = onExit;

            if (!gameObject.activeInHierarchy)
                gameObject.SetActive(true);

            if (_popupRoot != null)
                _popupRoot.SetActive(true);

            if (_titleText != null)
                _titleText.text = string.IsNullOrEmpty(title) ? _defaultTitle : title;

            if (_exitButton != null)
                _exitButton.gameObject.SetActive(showExitButton);

            // 이전 입력 초기화 후 포커스
            if (_inputField != null)
            {
                _inputField.text = initialValue ?? string.Empty;
                _inputField.ActivateInputField();
            }

            FadeTo(1f);
        }

        /// <summary>(외부 호출) 콜백 실행 없이 입력창을 즉시 닫는다. (가이드 스킵 등)</summary>
        public void Close()
        {
            _pendingCallback = null;
            _pendingExit     = null;
            Hide();
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private void OnSubmit()
        {
            var input = _inputField != null ? _inputField.text.Trim() : string.Empty;

            if (string.IsNullOrEmpty(input))
                return;

            Hide();
            var callback = _pendingCallback;
            _pendingCallback = null;
            _pendingExit     = null;
            callback?.Invoke(input);
        }

        // 나가기 버튼: 입력을 제출하지 않고 창을 닫은 뒤 onExit 콜백을 호출한다.
        private void OnExit()
        {
            var onExit = _pendingExit;
            _pendingCallback = null;
            _pendingExit     = null;
            Hide();
            onExit?.Invoke();
        }

        private void Hide()
        {
            FadeTo(0f);
            if (_popupRoot != null)
                _popupRoot.SetActive(false);
        }

        private void FadeTo(float target)
        {
            if (_fadeRoutine != null)
                StopCoroutine(_fadeRoutine);

            _fadeRoutine = FadeRoutine(target);
            StartCoroutine(_fadeRoutine);
        }

        private System.Collections.IEnumerator FadeRoutine(float target)
        {
            if (_canvasGroup == null) yield break;

            float start   = _canvasGroup.alpha;
            float elapsed = 0f;

            while (elapsed < _fadeDuration)
            {
                elapsed           += Time.deltaTime;
                _canvasGroup.alpha = Mathf.Lerp(start, target, elapsed / _fadeDuration);
                yield return null;
            }

            _canvasGroup.alpha = target;
        }
    }
}
