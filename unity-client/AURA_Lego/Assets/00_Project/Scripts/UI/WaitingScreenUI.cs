using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using LegoTwin.Managers;

namespace LegoTwin.UI
{
    /// <summary>
    /// 대기화면 UI (영상 → 입장 버튼 게이트 방식).
    ///
    /// 흐름:
    ///   씬 시작 → Show() (대기화면 + 영상 재생 + 대기 텍스트 페이드 인, 입장 버튼 숨김)
    ///   세션 준비 완료 → GameFlowManager가 ShowEnterButton(onEnter) 호출
    ///     → (Mock) _mockPrepSeconds 동안 영상 유지 후 / (Server) 즉시
    ///       영상 페이드 아웃 + 대기 텍스트 숨김 + 준비 텍스트·입장 버튼 표시
    ///   플레이어가 입장 버튼 클릭 → onEnter 실행(가이드 시작) → Hide()
    ///
    /// 종료 → 대기화면 복귀(A안): 종료 시 씬 리로드 → 새 씬에서 이 컴포넌트가 다시 생성되며
    ///   Awake에서 자동 표시된다(영상도 Play On Awake로 처음부터 재생).
    ///
    /// 텍스트 분리:
    ///   _waitingText = 영상과 함께(대기 중) 표시(페이드 인). _readyText = 영상 꺼진 뒤(입장 버튼과 함께) 표시.
    ///   두 텍스트는 별도 GameObject라 내용·위치·스타일을 각각 자유롭게 구성한다.
    ///
    /// Mock 모드 시뮬레이션:
    ///   Mock은 세션이 즉시 로드돼 ShowEnterButton이 바로 호출된다. 실제 전시(Server)의
    ///   "준비 대기" 체감을 테스트하려고, Mock에서는 _mockPrepSeconds 만큼 영상을 더 보여준 뒤
    ///   영상을 끄고 입장 버튼을 노출한다. Server에서는 실제 준비 시점(ShowEnterButton)에 즉시 노출.
    ///
    /// Hierarchy 예:
    ///   WaitingScreen (Canvas — Overlay, Sort Order 높게)   ← 항상 활성, 이 컴포넌트 부착
    ///     └── Root (CanvasGroup, 활성으로 시작)
    ///          ├── VideoRoot (RawImage + VideoPlayer)  ← 대기 중 영상, 준비되면 페이드 아웃
    ///          ├── WaitingText (TMP)   ← 영상과 함께 표시(페이드 인), 활성 시작
    ///          ├── ReadyText   (TMP)   ← 영상 OFF 후 표시, 비활성 시작
    ///          └── EnterButton (Button — "입장", 비활성으로 시작)
    ///
    /// 체크리스트:
    ///   [ ] _root        → Root (활성 시작)
    ///   [ ] _canvasGroup → Root의 CanvasGroup (페이드용, 선택)
    ///   [ ] _videoRoot   → VideoRoot (영상 RawImage 오브젝트). 준비되면 페이드 후 SetActive(false)
    ///   [ ] _waitingText → WaitingText (영상과 함께 표시, 활성 시작)
    ///   [ ] _readyText   → ReadyText   (영상 꺼진 뒤 표시, 비활성 시작)
    ///   [ ] _enterButton → EnterButton (없으면 게이트 없이 즉시 진행)
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

        [Header("대기 영상")]
        [Tooltip("대기 중 재생할 영상 오브젝트(RawImage+VideoPlayer). 준비되면 페이드 아웃 후 꺼진다.")]
        [SerializeField] private GameObject _videoRoot;

        [Tooltip("영상이 사라질 때 페이드 아웃 시간(초). 0이면 즉시 꺼짐.")]
        [SerializeField] private float _videoFadeDuration = 1f;

        [Header("안내 텍스트 (영상 ON/OFF로 구분)")]
        [Tooltip("영상과 함께(대기 중) 표시할 텍스트 오브젝트. 표시될 때 서서히 나타난다(페이드 인).")]
        [SerializeField] private GameObject _waitingText;

        [Tooltip("대기 텍스트가 나타나는 페이드 인 시간(초). 0이면 즉시 표시.")]
        [SerializeField] private float _waitingTextFadeDuration = 0.8f;

        [Tooltip("영상이 꺼진 뒤(입장 버튼과 함께) 표시할 텍스트 오브젝트. 대기 중에는 숨김(비활성 시작).")]
        [SerializeField] private GameObject _readyText;

        [Header("입장 버튼")]
        [Tooltip("세션 준비 완료 시 활성화될 입장 버튼. 대기 중에는 비활성으로 시작.")]
        [SerializeField] private Button _enterButton;

        [Header("Mock 시뮬레이션")]
        [Tooltip("Mock 모드 전용: 세션이 즉시 준비돼도 이 시간(초)만큼 영상을 더 보여준 뒤\n" +
                 "영상을 끄고 입장 버튼을 노출한다. Server 모드에서는 무시(실제 준비 시점에 노출).")]
        [SerializeField] private float _mockPrepSeconds = 5f;

        private Coroutine   _fade;
        private Coroutine   _reveal;
        private Coroutine   _videoFade;
        private Coroutine   _waitingTextFade;
        private CanvasGroup _videoCg;
        private CanvasGroup _waitingTextCg;
        private float       _shownTime;
        private Action      _onEnter;

        private void Awake()
        {
            if (_enterButton != null)
            {
                _enterButton.onClick.AddListener(OnEnterPressed);
                _enterButton.gameObject.SetActive(false);
            }

            // 페이드용 CanvasGroup 확보(없으면 추가). Show()보다 먼저 준비.
            _videoCg       = EnsureCanvasGroup(_videoRoot);
            _waitingTextCg = EnsureCanvasGroup(_waitingText);

            // ShowEnterButton이 다른 스크립트의 Start에서 먼저 불릴 수 있어, 표시 상태를
            // Awake에서 확정해 둔다(모든 Start보다 Awake가 먼저 실행됨).
            Show();
        }

        private void OnDestroy()
        {
            if (_enterButton != null)
                _enterButton.onClick.RemoveListener(OnEnterPressed);
        }

        // ── 공개 API ─────────────────────────────────────────────────

        /// <summary>
        /// 세션 준비 완료 시 호출(GameFlowManager).
        /// Mock: _mockPrepSeconds 경과까지 영상 유지 후 입장 노출. Server: 즉시 노출.
        /// 입장 버튼이 미연결이면 게이트 없이 즉시 onEnter 실행(폴백).
        /// </summary>
        public void ShowEnterButton(Action onEnter)
        {
            _onEnter = onEnter;

            bool isMock = SessionManager.Instance != null
                       && SessionManager.Instance.dataSourceMode == DataSourceMode.Mock;
            float remaining = isMock
                ? Mathf.Max(0f, _mockPrepSeconds - (Time.unscaledTime - _shownTime))
                : 0f;

            if (_reveal != null) StopCoroutine(_reveal);
            if (remaining > 0f)
                _reveal = StartCoroutine(RevealAfterDelay(remaining));
            else
                RevealReady();
        }

        /// <summary>대기화면 표시 (영상 ON + 대기 텍스트 페이드 인, 준비 텍스트·입장 버튼 숨김).</summary>
        public void Show()
        {
            _shownTime = Time.unscaledTime;

            if (_videoFade != null) { StopCoroutine(_videoFade); _videoFade = null; }
            if (_videoRoot != null)
            {
                if (_videoCg != null) _videoCg.alpha = 1f;   // 페이드 알파 복구
                _videoRoot.SetActive(true);                  // 영상 재생(Play On Awake)
            }

            // 대기 텍스트: 활성화 후 알파 0 → 1 페이드 인
            if (_waitingTextFade != null) { StopCoroutine(_waitingTextFade); _waitingTextFade = null; }
            if (_waitingText != null)
            {
                _waitingText.SetActive(true);
                if (_waitingTextCg != null && _waitingTextFadeDuration > 0f)
                {
                    _waitingTextCg.alpha = 0f;
                    _waitingTextFade = StartCoroutine(FadeCanvasGroup(_waitingTextCg, 1f, _waitingTextFadeDuration));
                }
                else if (_waitingTextCg != null)
                {
                    _waitingTextCg.alpha = 1f;
                }
            }

            if (_readyText   != null) _readyText.SetActive(false);   // 준비 텍스트 OFF
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

        // 준비 완료 표시: 영상 페이드 아웃 + 대기 텍스트 끄고, 준비 텍스트·입장 버튼 노출.
        private void RevealReady()
        {
            FadeOutVideo();                                          // 영상 천천히 사라짐

            if (_waitingTextFade != null) { StopCoroutine(_waitingTextFade); _waitingTextFade = null; }
            if (_waitingText != null) _waitingText.SetActive(false); // 대기 텍스트 OFF
            if (_readyText   != null) _readyText.SetActive(true);    // 준비 텍스트 ON

            if (_enterButton != null)
            {
                _enterButton.gameObject.SetActive(true);
                _enterButton.interactable = true;
            }
            else
            {
                OnEnterPressed();
            }
        }

        private IEnumerator RevealAfterDelay(float delay)
        {
            yield return new WaitForSecondsRealtime(delay);
            _reveal = null;
            RevealReady();
        }

        // 영상을 페이드 아웃 후 비활성화(디코딩 정지). CanvasGroup 없거나 시간 0이면 즉시 끔.
        private void FadeOutVideo()
        {
            if (_videoRoot == null) return;
            if (_videoCg == null || _videoFadeDuration <= 0f)
            {
                _videoRoot.SetActive(false);
                return;
            }
            if (_videoFade != null) StopCoroutine(_videoFade);
            _videoFade = StartCoroutine(VideoFadeOutRoutine());
        }

        private IEnumerator VideoFadeOutRoutine()
        {
            yield return FadeCanvasGroup(_videoCg, 0f, _videoFadeDuration);
            _videoRoot.SetActive(false);   // 페이드 끝나면 끔(영상·디코딩 정지)
            _videoFade = null;
        }

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
            _fade = StartCoroutine(RootFadeRoutine(targetAlpha, deactivateAtEnd));
        }

        private IEnumerator RootFadeRoutine(float targetAlpha, bool deactivateAtEnd)
        {
            yield return FadeCanvasGroup(_canvasGroup, targetAlpha, _fadeDuration);
            if (deactivateAtEnd && _root != null) _root.SetActive(false);
            _fade = null;
        }

        // 공용 CanvasGroup 알파 페이드 (타임스케일 무관).
        // 씬 로드 직후 첫 프레임은 unscaledDeltaTime이 수 초로 튀어(로딩 hitch) 페이드가 한 번에
        // 끝나버린다. 프레임당 진행량을 상한(MaxStep)으로 clamp해 항상 부드럽게 재생되게 한다.
        private const float MaxStep = 0.05f;

        private static IEnumerator FadeCanvasGroup(CanvasGroup cg, float targetAlpha, float duration)
        {
            float startAlpha = cg.alpha;
            float t = 0f;
            while (t < duration)
            {
                t += Mathf.Min(Time.unscaledDeltaTime, MaxStep);
                cg.alpha = Mathf.Lerp(startAlpha, targetAlpha, t / duration);
                yield return null;
            }
            cg.alpha = targetAlpha;
        }

        // GameObject에서 CanvasGroup을 찾고 없으면 추가. null이면 null 반환.
        private static CanvasGroup EnsureCanvasGroup(GameObject go)
        {
            if (go == null) return null;
            var cg = go.GetComponent<CanvasGroup>();
            if (cg == null) cg = go.AddComponent<CanvasGroup>();
            return cg;
        }
    }
}
