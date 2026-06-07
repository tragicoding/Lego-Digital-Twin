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
    ///               └── SubmitButton (Button)
    ///                    └── ButtonText (TMP_Text — "확인")
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] _popupRoot   — PopupRoot GameObject 연결
    ///   [ ] _inputField  — TMP_InputField 연결
    ///   [ ] _submitButton — Button 연결
    ///   [ ] _canvasGroup — PopupRoot 의 CanvasGroup 연결
    ///   [ ] GameFlowManager._motionPromptUI 에 이 GameObject 드래그
    /// </summary>
    public class MotionPromptUI : MonoBehaviour
    {
        [Header("UI 요소")]
        [SerializeField] private GameObject     _popupRoot;
        [SerializeField] private TMP_InputField _inputField;
        [SerializeField] private Button         _submitButton;
        [SerializeField] private CanvasGroup    _canvasGroup;

        [Header("연출 설정")]
        [SerializeField] private float _fadeDuration = 0.2f;

        private Action<string> _pendingCallback;
        private System.Collections.IEnumerator _fadeRoutine;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Awake()
        {
            if (_popupRoot != null)
                _popupRoot.SetActive(false);

            if (_canvasGroup != null)
                _canvasGroup.alpha = 0f;

            _submitButton?.onClick.AddListener(OnSubmit);
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
            _pendingCallback = callback;

            if (!gameObject.activeInHierarchy)
                gameObject.SetActive(true);

            if (_popupRoot != null)
                _popupRoot.SetActive(true);

            // 이전 입력 초기화 후 포커스
            if (_inputField != null)
            {
                _inputField.text = string.Empty;
                _inputField.ActivateInputField();
            }

            FadeTo(1f);
        }

        /// <summary>(외부 호출) 콜백 실행 없이 입력창을 즉시 닫는다. (가이드 스킵 등)</summary>
        public void Close()
        {
            _pendingCallback = null;
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
            _pendingCallback?.Invoke(input);
            _pendingCallback = null;
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
