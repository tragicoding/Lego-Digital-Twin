using System.Collections;
using TMPro;
using UnityEngine;

namespace LegoTwin.UI
{
    /// <summary>
    /// 자유 모드 전환 팝업 UI.
    ///
    /// 역할:
    ///   - OnFreeModeSwitched 이벤트 발생 시 "자유모드로 전환되었습니다." 페이드 인 표시
    ///   - displayDuration 초 후 자동으로 페이드 아웃
    ///
    /// 씬 Hierarchy 구조:
    ///   FreeModePopupUI         (Canvas)
    ///     └── PopupRoot         (GameObject + CanvasGroup — 페이드 대상)
    ///          └── PopupPanel   (Image — 배경 패널)
    ///               └── PopupText  (TMP_Text — 안내 문구)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] _popupRoot   — PopupRoot GameObject 연결
    ///   [ ] _popupText   — PopupText TMP_Text 연결 (선택, 비워두면 기본 문구 사용)
    ///   [ ] _canvasGroup — PopupRoot 의 CanvasGroup 연결
    ///   [ ] GameFlowManager._freeModePopup 에 이 GameObject 드래그
    ///       → Show() / Hide() 를 직접 호출하려면 GameFlowManager 에서
    ///         GetComponent<FreeModePopupUI>() 로 참조 후 사용
    /// </summary>
    public class FreeModePopupUI : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("UI 요소")]
        [Tooltip("Show/Hide 할 루트 오브젝트 (PopupRoot)")]
        [SerializeField] private GameObject  _popupRoot;

        [Tooltip("안내 문구 텍스트 (비워두면 기본값 사용)")]
        [SerializeField] private TMP_Text    _popupText;

        [Tooltip("페이드 인/아웃용 CanvasGroup (PopupRoot 에 추가)")]
        [SerializeField] private CanvasGroup _canvasGroup;

        [Header("연출 설정")]
        [Tooltip("페이드 인/아웃 시간 (초)")]
        [SerializeField] private float _fadeDuration   = 0.4f;

        [Tooltip("팝업이 표시되는 시간 (초) — 이 시간 후 자동으로 페이드 아웃")]
        [SerializeField] private float _displayDuration = 2.0f;

        [Tooltip("표시할 안내 문구")]
        [SerializeField] private string _message = "자유모드로 전환되었습니다.";

        // ── 런타임 ───────────────────────────────────────────────────
        private Coroutine _routine;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Awake()
        {
            if (_popupRoot != null)
                _popupRoot.SetActive(false);

            if (_canvasGroup != null)
                _canvasGroup.alpha = 0f;
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 팝업을 표시한다.
        /// 페이드 인 → displayDuration 대기 → 페이드 아웃 → 비활성화 자동 실행.
        /// GameFlowManager._freeModePopup.SetActive(true) 대신 이 메서드를 직접 호출해도 된다.
        /// </summary>
        public void Show()
        {
            if (_popupText != null)
                _popupText.text = _message;

            // Canvas(부모) 자체가 비활성이면 먼저 활성화 — 비활성 상태에서는 StartCoroutine 불가
            if (!gameObject.activeInHierarchy)
                gameObject.SetActive(true);

            if (_popupRoot != null)
                _popupRoot.SetActive(true);

            if (_routine != null)
                StopCoroutine(_routine);

            _routine = StartCoroutine(ShowRoutine());
        }

        /// <summary>즉시 숨긴다 (페이드 아웃 후 비활성화).</summary>
        public void Hide()
        {
            if (_routine != null)
                StopCoroutine(_routine);

            _routine = StartCoroutine(HideRoutine());
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private IEnumerator ShowRoutine()
        {
            yield return FadeRoutine(1f);                      // 페이드 인
            yield return new WaitForSeconds(_displayDuration); // 노출 유지
            yield return HideRoutine();                        // 페이드 아웃 + 비활성화
        }

        private IEnumerator HideRoutine()
        {
            yield return FadeRoutine(0f);

            if (_popupRoot != null)
                _popupRoot.SetActive(false);
        }

        private IEnumerator FadeRoutine(float target)
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
