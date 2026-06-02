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
        [SerializeField] private CanvasGroup _canvasGroup;

        [Header("연출 설정")]
        [SerializeField] private float  _fadeDuration = 0.2f;
        [SerializeField] private string _message      = "시그니처 동작으로 설정하시겠습니까?";

        private Action<bool> _pendingCallback;
        private Coroutine    _fadeRoutine;

        private void Awake()
        {
            if (_popupRoot  != null) _popupRoot.SetActive(false);
            if (_canvasGroup != null) _canvasGroup.alpha = 0f;

            _yesButton?.onClick.AddListener(() => Confirm(true));
            _noButton?.onClick.AddListener(()  => Confirm(false));
        }

        private void OnDestroy()
        {
            _yesButton?.onClick.RemoveAllListeners();
            _noButton?.onClick.RemoveAllListeners();
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 확인 다이얼로그를 표시한다.
        /// 예 → callback(true), 아니오 → callback(false).
        /// </summary>
        public void Show(Action<bool> callback)
        {
            _pendingCallback = callback;

            if (_messageText != null) _messageText.text = _message;
            if (!gameObject.activeInHierarchy) gameObject.SetActive(true);
            if (_popupRoot   != null) _popupRoot.SetActive(true);

            FadeTo(1f);
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private void Confirm(bool yes)
        {
            if (_popupRoot  != null) _popupRoot.SetActive(false);
            if (_canvasGroup != null) _canvasGroup.alpha = 0f;

            var cb = _pendingCallback;
            _pendingCallback = null;
            cb?.Invoke(yes);
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
