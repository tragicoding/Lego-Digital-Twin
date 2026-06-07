using System;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace LegoTwin.UI
{
    /// <summary>
    /// 시그니처 동작 설정 확인 UI.
    ///
    /// 동작 흐름:
    ///   GuideNPCController Step 3 (모션 확인 후)
    ///     → GameFlowManager.OnSignatureMotionConfirmRequested
    ///       → Show(callback)
    ///         → 예 버튼: callback(true)  → 서버 저장
    ///         → 아니오 버튼: callback(false) → 생략
    ///
    /// 씬 Hierarchy 구조:
    ///   SignatureMotionConfirmUI    (Canvas — Screen Space Overlay)
    ///     └── PopupRoot             (GameObject + CanvasGroup — 비활성으로 시작)
    ///          └── Panel            (Image — 배경)
    ///               ├── MessageText (TMP_Text — "시그니처 동작으로 설정하시겠습니까?")
    ///               ├── YesButton   (Button — "예")
    ///               └── NoButton    (Button — "아니오")
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Canvas: Render Mode → Screen Space - Overlay
    ///   [ ] _popupRoot   — PopupRoot 연결 (비활성으로 시작)
    ///   [ ] _messageText — MessageText TMP_Text 연결
    ///   [ ] _yesButton   — YesButton Button 연결
    ///   [ ] _noButton    — NoButton Button 연결
    ///   [ ] _canvasGroup — PopupRoot의 CanvasGroup 연결
    ///   [ ] GameFlowManager._signatureConfirmUI 에 이 GameObject 드래그
    /// </summary>
    public class SignatureMotionConfirmUI : MonoBehaviour
    {
        [Header("UI 요소")]
        [SerializeField] private GameObject  _popupRoot;
        [SerializeField] private TMP_Text    _messageText;
        [SerializeField] private Button      _yesButton;
        [SerializeField] private Button      _noButton;
        [Tooltip("자유 모드에서만 표시. 안내 모드에서는 Show() 호출 시 자동으로 숨김.")]
        [SerializeField] private Button      _exitButton;
        [Tooltip("아니오 버튼 아래 보조 문구 (NoButton 자식 캡션). 미연결 시 캡션 변경 안 함.")]
        [SerializeField] private TMP_Text    _noCaptionText;
        [SerializeField] private CanvasGroup _canvasGroup;

        [Header("연출 설정")]
        [SerializeField] private float  _fadeDuration = 0.2f;
        [SerializeField] private string _message      = "현재 동작을 \n시그니처 동작으로 설정할까요?";

        private Action<bool> _pendingCallback;
        private Action       _pendingOnExit;
        private Coroutine    _fadeRoutine;
        private string       _defaultNoCaption;   // 아니오 캡션 기본값 (복원용)

        private void Awake()
        {
            if (_popupRoot     != null) _popupRoot.SetActive(false);
            if (_canvasGroup   != null) _canvasGroup.alpha = 0f;
            if (_exitButton    != null) _exitButton.gameObject.SetActive(false);

            // _noCaptionText 미연결 시 NoButton 자식에서 보조 캡션 TMP 자동 탐색.
            // NoButton에는 "아니오" 라벨 + 보조 캡션("다른 동작도 해볼래요!") 2개의 TMP가 있으므로
            // 더 긴 텍스트 쪽을 캡션으로 추정한다.
            if (_noCaptionText == null && _noButton != null)
            {
                var texts = _noButton.GetComponentsInChildren<TMP_Text>(true);
                if (texts.Length >= 2)
                {
                    TMP_Text longest = texts[0];
                    foreach (var t in texts)
                        if ((t.text?.Length ?? 0) > (longest.text?.Length ?? 0)) longest = t;
                    _noCaptionText = longest;
                }
            }
            if (_noCaptionText != null) _defaultNoCaption = _noCaptionText.text;

            _yesButton?.onClick.AddListener(() => Confirm(true));
            _noButton?.onClick.AddListener(()  => Confirm(false));
            _exitButton?.onClick.AddListener(Exit);
        }

        private void OnDestroy()
        {
            _yesButton?.onClick.RemoveAllListeners();
            _noButton?.onClick.RemoveAllListeners();
            _exitButton?.onClick.RemoveAllListeners();
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 확인 다이얼로그를 표시한다.
        /// 예 → callback(true) / 아니오 → callback(false) / 나가기 → onExit().
        /// showExitButton: true 이면 나가기 버튼 표시 (자유 모드), false 이면 숨김 (안내 모드).
        /// message: null 이면 Inspector 기본 메시지 사용, 값 전달 시 해당 메시지로 덮어씀.
        /// noCaption: 아니오 버튼 아래 보조 문구. null 이면 기본 캡션으로 복원, 값 전달 시 덮어씀.
        /// </summary>
        public void Show(Action<bool> callback, bool showExitButton = false, Action onExit = null,
                         string message = null, string noCaption = null)
        {
            _pendingCallback = callback;
            _pendingOnExit   = onExit;

            if (_messageText   != null) _messageText.text   = message ?? _message;
            if (_noCaptionText != null) _noCaptionText.text = noCaption ?? _defaultNoCaption;
            if (_exitButton    != null) _exitButton.gameObject.SetActive(showExitButton);
            if (!gameObject.activeInHierarchy) gameObject.SetActive(true);
            if (_popupRoot   != null) _popupRoot.SetActive(true);

            FadeTo(1f);
        }

        /// <summary>(외부 호출) 콜백 실행 없이 다이얼로그를 즉시 닫는다. (가이드 스킵 등)</summary>
        public void Close()
        {
            _pendingCallback = null;
            _pendingOnExit   = null;
            if (_popupRoot   != null) _popupRoot.SetActive(false);
            if (_canvasGroup != null) _canvasGroup.alpha = 0f;
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private void Confirm(bool yes)
        {
            if (_popupRoot   != null) _popupRoot.SetActive(false);
            if (_canvasGroup != null) _canvasGroup.alpha = 0f;

            var cb = _pendingCallback;
            _pendingCallback = null;
            _pendingOnExit   = null;
            cb?.Invoke(yes);
        }

        private void Exit()
        {
            if (_popupRoot   != null) _popupRoot.SetActive(false);
            if (_canvasGroup != null) _canvasGroup.alpha = 0f;

            var cb = _pendingOnExit;
            _pendingCallback = null;
            _pendingOnExit   = null;
            cb?.Invoke();
        }

        private void FadeTo(float target)
        {
            if (_fadeRoutine != null) StopCoroutine(_fadeRoutine);
            _fadeRoutine = StartCoroutine(FadeRoutine(target));
        }

        private IEnumerator FadeRoutine(float target)
        {
            if (_canvasGroup == null) yield break;
            float start = _canvasGroup.alpha, elapsed = 0f;
            while (elapsed < _fadeDuration)
            {
                elapsed           += Time.deltaTime;
                _canvasGroup.alpha  = Mathf.Lerp(start, target, elapsed / _fadeDuration);
                yield return null;
            }
            _canvasGroup.alpha = target;
        }
    }
}
