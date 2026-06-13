using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace LegoTwin.UI
{
    /// <summary>
    /// 대기화면 UI (입장 버튼 게이트 방식).
    ///
    /// 흐름:
    ///   씬 시작 → Show() (대기화면 표시, 입장 버튼 숨김)
    ///   세션 준비 완료 → GameFlowManager가 ShowEnterButton(onEnter) 호출 → 입장 버튼 활성화
    ///   플레이어가 입장 버튼 클릭 → onEnter 실행(가이드 시작) → Hide()
    ///
    /// 종료 → 대기화면 복귀(A안): 종료 시 씬 리로드 → 새 씬에서 이 컴포넌트가 다시 생성되며
    ///   Start에서 자동 표시된다(별도 "다시 표시" 호출 불필요).
    ///
    /// 의존성: 없음. 외부와는 Action 콜백으로만 통신하는 순수 뷰다(낮은 결합).
    ///   GameFlowManager·SessionManager·PlazaManager를 참조하지 않는다 → Mock·Server 무관 동일.
    ///
    /// Hierarchy 예:
    ///   WaitingScreen (Canvas — Overlay, Sort Order 높게)   ← 항상 활성, 이 컴포넌트 부착
    ///     └── Root (CanvasGroup, 활성으로 시작)
    ///          ├── Background (Image, 불투명)
    ///          ├── StatusText (TMP — "관람객을 기다리는 중…")
    ///          └── EnterButton (Button — "입장", 비활성으로 시작)
    ///
    /// 체크리스트:
    ///   [ ] _root        → Root (활성 시작)
    ///   [ ] _canvasGroup → Root의 CanvasGroup (페이드용, 선택)
    ///   [ ] _enterButton → EnterButton (없으면 게이트 없이 즉시 진행)
    ///   [ ] _statusText  → StatusText (선택 — 준비 전/후 문구 전환)
    /// </summary>
    public class WaitingScreenUI : MonoBehaviour
    {
        [Header("UI")]
        [Tooltip("대기화면 본체. 활성으로 시작하고, 입장 후 숨겨진다.")]
        [SerializeField] private GameObject _root;

        [Tooltip("페이드 인/아웃용 CanvasGroup (선택). 없으면 즉시 토글.")]
        [SerializeField] private CanvasGroup _canvasGroup;

        [SerializeField] private float _fadeDuration = 0.4f;

        [Tooltip("최소 표시 시간(초). 입장 직후에도 한 프레임만 번쩍이지 않도록 보장.")]
        [SerializeField] private float _minimumShowSeconds = 0.6f;

        [Header("입장 버튼")]
        [Tooltip("세션 준비 완료 시 활성화될 입장 버튼. 대기 중에는 비활성으로 시작.")]
        [SerializeField] private Button _enterButton;

        [Tooltip("준비 전/후 안내 문구를 바꿀 TMP (선택).")]
        [SerializeField] private TMP_Text _statusText;
        [SerializeField] private string _waitingMessage = "관람객을 기다리는 중…";
        [SerializeField] private string _readyMessage   = "준비 완료! 입장하세요";

        private Coroutine _fade;
        private float     _shownTime;
        private Action    _onEnter;

        private void Awake()
        {
            if (_enterButton != null)
            {
                _enterButton.onClick.AddListener(OnEnterPressed);
                _enterButton.gameObject.SetActive(false);
            }
        }

        private void OnDestroy()
        {
            if (_enterButton != null)
                _enterButton.onClick.RemoveListener(OnEnterPressed);
        }

        private void Start() => Show();

        // ── 공개 API ─────────────────────────────────────────────────

        /// <summary>
        /// 세션 준비 완료 시 호출(GameFlowManager) — 입장 버튼을 활성화하고 클릭 시 실행할 콜백을 등록한다.
        /// 입장 버튼이 미연결이면 게이트 없이 즉시 onEnter 실행(폴백).
        /// </summary>
        public void ShowEnterButton(Action onEnter)
        {
            _onEnter = onEnter;
            if (_statusText != null) _statusText.text = _readyMessage;

            if (_enterButton != null)
            {
                _enterButton.gameObject.SetActive(true);
                _enterButton.interactable = true;
            }
            else
            {
                OnEnterPressed();   // 버튼 없으면 즉시 진행
            }
        }

        /// <summary>대기화면 표시 (입장 버튼은 숨김 상태로 시작).</summary>
        public void Show()
        {
            _shownTime = Time.unscaledTime;
            if (_statusText  != null) _statusText.text = _waitingMessage;
            if (_enterButton != null) _enterButton.gameObject.SetActive(false);
            if (_root != null) _root.SetActive(true);
            BeginFade(1f, deactivateAtEnd: false);
        }

        /// <summary>대기화면 숨김 (최소 표시 시간 보장 후 페이드 아웃·비활성).</summary>
        public void Hide()
        {
            float remaining = _minimumShowSeconds - (Time.unscaledTime - _shownTime);
            if (remaining > 0f)
            {
                if (_fade != null) StopCoroutine(_fade);
                _fade = StartCoroutine(HideAfterDelay(remaining));
            }
            else
            {
                BeginFade(0f, deactivateAtEnd: true);
            }
        }

        // ── 내부 ─────────────────────────────────────────────────────

        private void OnEnterPressed()
        {
            if (_enterButton != null) _enterButton.interactable = false;   // 중복 클릭 방지
            var cb = _onEnter;
            _onEnter = null;
            cb?.Invoke();
            Hide();
        }

        private IEnumerator HideAfterDelay(float delay)
        {
            yield return new WaitForSecondsRealtime(delay);
            _fade = null;   // 자기 자신을 StopCoroutine 하지 않도록 먼저 비움
            BeginFade(0f, deactivateAtEnd: true);
        }

        private void BeginFade(float targetAlpha, bool deactivateAtEnd)
        {
            if (_canvasGroup == null)
            {
                if (deactivateAtEnd && _root != null) _root.SetActive(false);
                return;
            }

            if (_fade != null) StopCoroutine(_fade);
            _fade = StartCoroutine(FadeRoutine(targetAlpha, deactivateAtEnd));
        }

        private IEnumerator FadeRoutine(float targetAlpha, bool deactivateAtEnd)
        {
            float startAlpha = _canvasGroup.alpha;
            float t = 0f;
            while (t < _fadeDuration)
            {
                t += Time.unscaledDeltaTime;
                _canvasGroup.alpha = Mathf.Lerp(startAlpha, targetAlpha, t / _fadeDuration);
                yield return null;
            }
            _canvasGroup.alpha = targetAlpha;
            if (deactivateAtEnd && _root != null) _root.SetActive(false);
            _fade = null;
        }
    }
}
