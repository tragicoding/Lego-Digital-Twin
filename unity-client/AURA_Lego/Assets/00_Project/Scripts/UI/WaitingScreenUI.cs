using System;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using LegoTwin.Managers;
using LegoTwin.Core;

namespace LegoTwin.UI
{
    /// <summary>
    /// 대기화면 UI — 대기 영상 → 이름 입력 → 연동 로딩 영상 → 시작 영상(입장) 게이트.
    ///
    /// 화면 순서:
    ///   ⓪ 대기(attract) 영상  (_attractVideoRoot) — 클릭/터치/키 입력 시 ①로 (선택)
    ///   ① 캐릭터 이름 입력  (텍스트 + InputField + "next")
    ///   ② 오브제   이름 입력  (텍스트 + InputField + "next")
    ///   ③ 연동 로딩 영상     (_videoRoot, 예: 0614_2) — 스폰/로드 동안 표시
    ///   ④ 시작 영상 + 입장   (_enterVideoRoot, 예: 0614_1) + "START" 버튼
    ///
    /// 외부와의 통신은 콜백 두 개뿐이라 순수 뷰로 남는다(GameFlow/Session/스폰 미참조):
    ///   - BeginNameEntry(onNamesEntered): ②까지 입력 완료 시 (캐릭터명, 오브제명) 전달 → ③로 전환
    ///   - ShowEnterButton(onEnter):       세션·캐릭터 준비 완료 시 ③→④ 전환, "START" 클릭 시 onEnter
    /// 입력값을 어디에 저장할지(SessionManager.SetNames 등)·언제 스폰할지는 GameFlowManager가 결정한다.
    /// 모든 단계가 세션/서버를 참조하지 않으므로 Mock·Server에서 화면 흐름이 100% 동일하다.
    ///
    /// 게이트 의미:
    ///   이름 입력(①②)이 끝나야 GameFlowManager가 이름을 적용하고 스폰을 시작한다(③에서 로드).
    ///   → 가이드 NPC·내 창작물이 처음부터 입력한 이름으로 생성된다(사후 갱신 불필요).
    ///   스폰이 끝나면(가이드 준비) ShowEnterButton 으로 ④ 시작 영상 + "START" 가 노출된다.
    ///
    /// 종료 → 대기화면 복귀(A안): 종료 시 씬 리로드 → 새 씬에서 이 컴포넌트가 다시 생성되며 Awake에서 ⓪부터 다시 시작.
    ///
    /// Mock 시뮬레이션:
    ///   Mock은 스폰이 즉시 끝나 ShowEnterButton이 바로 호출된다. 실제 전시(Server)의 "연동 로딩" 체감을
    ///   위해, Mock에서는 ③ 로딩 영상을 _mockPrepSeconds 만큼 더 보여준 뒤 ④로 넘어간다. Server는 즉시 노출.
    ///
    /// 폴백(미연결 시에도 안전):
    ///   - 대기 영상 미연결 → ⓪를 건너뛰고 곧장 ① 이름 입력부터.
    ///   - 이름 패널 미연결 → 이름 입력을 건너뛰고 빈 이름을 보고(기존 이름 유지) 바로 ③로 간다.
    ///   - 대기 영상·이름 패널 모두 미연결 → 바로 ③로(도입 전 동작과 호환).
    ///   - 입장 버튼 미연결 → 게이트 없이 즉시 onEnter 실행.
    ///   - InputField 비움 → 빈 문자열 보고(GameFlowManager가 기존 이름 유지).
    ///
    /// 씬 구성 체크리스트:
    ///   [ ] _root → 대기화면 본체(활성 시작)
    ///   [ ] (선택) _canvasGroup → 페이드용 CanvasGroup
    ///   [ ] (선택) _attractVideoRoot → 대기 영상(RawImage+VideoPlayer, Loop) / (선택) _attractTitle → 그 위 타이틀 Text(페이드 인)
    ///   [ ] _characterNamePanel / _characterNameInput / _characterNameNext / (선택) _characterNameTitle
    ///   [ ] _objectNamePanel    / _objectNameInput    / _objectNameNext    / (선택) _objectNameTitle
    ///       (InputField 의 Placeholder 에 "예) 푸앙이" / "예) 원형관" 입력 — 코드 아님)
    ///       (_xxxNameTitle = 그 단계의 타이틀 Text 오브젝트. 표시될 때 Text만 페이드 인)
    ///   [ ] _videoRoot      → 연동 로딩 영상(RawImage+VideoPlayer, 0614_2). 이름 패널보다 '뒤 형제'(위 렌더)로 둬야
    ///                          입력 위로 페이드 인하며 자연스럽게 덮인다.
    ///   [ ] _enterVideoRoot → 시작 영상(RawImage+VideoPlayer, 0614_1) / (선택) _enterTitle → 그 위 타이틀 Text(페이드 인)
    ///   [ ] _enterButton    → "START" (없으면 즉시 진행)
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

        [Header("⓪ 대기(attract) 영상")]
        [Tooltip("이름 입력 전에 보여줄 대기 영상(RawImage+VideoPlayer). 화면 클릭/터치/키 입력이 들어오면 이름 입력으로 넘어간다.\n" +
                 "미연결 시 곧장 캐릭터 이름 입력부터 시작한다(이 단계는 선택).")]
        [SerializeField] private GameObject _attractVideoRoot;

        [Tooltip("대기 영상 위 타이틀 Text 오브젝트(예: \"화면을 누르세요\"). 대기 영상이 켜질 때 페이드 인한다(미연결 시 즉시 표시). 선택.")]
        [SerializeField] private GameObject _attractTitle;

        [Tooltip("대기 타이틀을 깜빡임(페이드 인↔아웃 반복)으로 표시한다. 끄면 한 번만 페이드 인.")]
        [SerializeField] private bool _attractTitleBlink = true;

        [Tooltip("깜빡이는 타이틀(대기·시작)이 완전히 보일 때·안 보일 때 각각 머무는 시간(초). 페이드 시간은 Title Fade Duration을 쓴다.")]
        [SerializeField] private float _titleBlinkHold = 0.6f;

        [Tooltip("대기 영상이 사라질 때 페이드 아웃 시간(초). 0이면 즉시.")]
        [SerializeField] private float _attractFadeDuration = 0.5f;

        [Tooltip("대기 영상 표시 직후 이 시간(초) 동안은 입력을 무시한다(씬 리로드 직후 잔여 입력으로 즉시 넘어가는 것 방지).")]
        [SerializeField] private float _attractInputGrace = 0.4f;

        [Header("① 캐릭터 이름 입력")]
        [Tooltip("캐릭터 이름 입력 패널(텍스트 + InputField + next). 비활성으로 시작(대기 영상 입력 후 표시).")]
        [SerializeField] private GameObject _characterNamePanel;
        [Tooltip("캐릭터 이름 단계의 타이틀 Text 오브젝트(예: \"당신의 캐릭터 이름을 지어주세요!\").\n" +
                 "패널이 표시될 때 이 Text만 페이드 인한다(미연결 시 즉시 표시). 선택.")]
        [SerializeField] private GameObject _characterNameTitle;
        [SerializeField] private TMP_InputField _characterNameInput;
        [Tooltip("다음 단계(오브제 이름)로 넘어가는 버튼.")]
        [SerializeField] private Button _characterNameNext;

        [Header("② 오브제 이름 입력")]
        [Tooltip("오브제 이름 입력 패널(텍스트 + InputField + next). 비활성으로 시작.")]
        [SerializeField] private GameObject _objectNamePanel;
        [Tooltip("오브제 이름 단계의 타이틀 Text 오브젝트(예: \"당신의 오브제 이름을 지어주세요!\").\n" +
                 "패널이 표시될 때 이 Text만 페이드 인한다(미연결 시 즉시 표시). 선택.")]
        [SerializeField] private GameObject _objectNameTitle;
        [SerializeField] private TMP_InputField _objectNameInput;
        [Tooltip("연동 로딩 영상으로 넘어가는 버튼.")]
        [SerializeField] private Button _objectNameNext;

        [Tooltip("모든 단계(대기·캐릭터·오브제·시작)의 타이틀 Text가 나타날 때(페이드 인) 걸리는 시간(초). 0이면 즉시.")]
        [SerializeField] private float _titleFadeDuration = 0.4f;

        [Header("③ 연동 로딩 영상")]
        [Tooltip("이름 입력 후 스폰/로드 동안 재생할 영상(RawImage+VideoPlayer, 예: 0614_2). 준비되면 페이드 아웃.")]
        [SerializeField] private GameObject _videoRoot;

        [Tooltip("로딩 영상의 페이드 인(이름 입력→로딩 전환)·페이드 아웃(준비 완료) 시간(초). 0이면 즉시.")]
        [SerializeField] private float _videoFadeDuration = 1f;

        [Header("④ 시작 영상 + 입장")]
        [Tooltip("준비 완료 시 표시할 시작 영상(RawImage+VideoPlayer, 예: 0614_1). 비활성으로 시작.")]
        [SerializeField] private GameObject _enterVideoRoot;

        [Tooltip("시작 영상 위 타이틀 Text 오브젝트(예: \"준비 완료! START를 눌러주세요\"). 시작 영상이 켜질 때 이 Text가 나타난다(미연결 시 즉시 표시). 선택.")]
        [SerializeField] private GameObject _enterTitle;

        [Tooltip("시작 타이틀을 깜빡임(페이드 인↔아웃 반복)으로 표시한다. 끄면 한 번만 페이드 인.")]
        [SerializeField] private bool _enterTitleBlink = true;

        [Tooltip("시작 영상이 나타날 때 페이드 인 시간(초). 0이면 즉시 표시.")]
        [SerializeField] private float _enterVideoFadeDuration = 0.6f;

        [Tooltip("세션 준비 완료 시 활성화될 입장 버튼(START). 대기 중에는 비활성으로 시작.")]
        [SerializeField] private Button _enterButton;

        [Tooltip("준비 완료(로딩 영상 페이드 시작) 후 START 버튼이 나타나기까지 지연 시간(초). 0이면 즉시.")]
        [SerializeField] private float _enterButtonDelay = 1f;

        [Header("Mock 시뮬레이션")]
        [Tooltip("Mock 모드 전용: 스폰이 즉시 끝나도 이 시간(초)만큼 연동 로딩 영상을 더 보여준 뒤\n" +
                 "시작 영상·입장 버튼을 노출한다. Server 모드에서는 무시(실제 준비 시점에 노출).")]
        [SerializeField] private float _mockPrepSeconds = 5f;

        private Coroutine   _fade;
        private Coroutine   _reveal;
        private Coroutine   _attractFade;
        private Coroutine   _videoFade;
        private Coroutine   _enterVideoFade;
        private Coroutine   _attractTitleFade;
        private Coroutine   _charTitleFade;
        private Coroutine   _objTitleFade;
        private Coroutine   _enterTitleFade;
        private Coroutine   _buttonDelay;
        private CanvasGroup _attractCg;
        private CanvasGroup _videoCg;
        private CanvasGroup _enterVideoCg;
        private CanvasGroup _attractTitleCg;
        private CanvasGroup _charTitleCg;
        private CanvasGroup _objTitleCg;
        private CanvasGroup _enterTitleCg;
        private bool        _attractActive;
        private float       _attractStartTime;
        private float       _shownTime;
        private float       _loadingStartTime;
        private string      _characterName = "";
        private string      _objectName    = "";
        private Action               _onEnter;
        private Action<string,string> _onNamesEntered;

        private void Awake()
        {
            if (_characterNameNext != null) _characterNameNext.onClick.AddListener(OnCharacterNameNext);
            if (_objectNameNext    != null) _objectNameNext.onClick.AddListener(OnObjectNameNext);
            if (_enterButton != null)
            {
                _enterButton.onClick.AddListener(OnEnterPressed);
                _enterButton.gameObject.SetActive(false);
            }

            // 페이드용 CanvasGroup 확보(없으면 추가). Show()보다 먼저 준비.
            _attractCg    = EnsureCanvasGroup(_attractVideoRoot);
            _videoCg      = EnsureCanvasGroup(_videoRoot);
            _enterVideoCg = EnsureCanvasGroup(_enterVideoRoot);
            _attractTitleCg = EnsureCanvasGroup(_attractTitle);      // 타이틀 Text 페이드 인용(대기·캐릭터·오브제·시작)
            _charTitleCg    = EnsureCanvasGroup(_characterNameTitle);
            _objTitleCg     = EnsureCanvasGroup(_objectNameTitle);
            _enterTitleCg   = EnsureCanvasGroup(_enterTitle);

            // 표시 상태를 Awake에서 확정(모든 Start보다 먼저 실행됨).
            Show();
        }

        private void OnDestroy()
        {
            if (_characterNameNext != null) _characterNameNext.onClick.RemoveListener(OnCharacterNameNext);
            if (_objectNameNext    != null) _objectNameNext.onClick.RemoveListener(OnObjectNameNext);
            if (_enterButton       != null) _enterButton.onClick.RemoveListener(OnEnterPressed);
        }

        private void Start()
        {
            // ⓪ 대기영상 단계 동안 헤드셋 착용 감지를 계속 열어둔다(미착용이어도 자동 데스크톱 확정 보류).
            // 착용되면 즉시 VR, 대기영상을 떠날 때(AdvanceFromAttract) 미착용이면 데스크톱으로 확정.
            // Awake 순서와 무관하도록 Start에서 호출한다(XRModeManager.Instance 보장). 매니저 없으면 null-safe no-op.
            // 대기영상이 없으면 게이트를 열지 않아 XRModeManager 기본(타임드) 판정이 그대로 동작한다(무회귀).
            if (_attractActive) XRModeManager.Instance?.BeginGatedResolve();
        }

        private void Update()
        {
            // ⓪ 대기 영상 단계: 아무 키·마우스 클릭·터치가 들어오면 이름 입력으로 넘어간다.
            if (_attractActive)
            {
                if (Time.unscaledTime - _attractStartTime < _attractInputGrace) return;  // 직후 잔여 입력 무시
                if (AnyPressThisFrame())
                    AdvanceFromAttract();
                return;   // 대기 단계에서는 아래 Enter(이름 next) 처리로 흘리지 않는다
            }

            // ①② 이름 입력 단계: Enter 키로도 next 버튼이 눌리게 한다(마우스 클릭과 병행).
            // 현재 활성인 이름 패널의 next만 실행하므로, 로딩·시작 단계에서는 영향 없다.
            // (MotionPromptUI와 동일한 Enter 폴링 패턴)
            if (!Input.GetKeyDown(KeyCode.Return) && !Input.GetKeyDown(KeyCode.KeypadEnter)) return;

            if (_characterNamePanel != null && _characterNamePanel.activeSelf)
                OnCharacterNameNext();
            else if (_objectNamePanel != null && _objectNamePanel.activeSelf)
                OnObjectNameNext();
        }

        // 이번 프레임에 키보드/마우스/터치 입력이 시작됐는지. (Input.anyKeyDown은 키보드+마우스 버튼 포함)
        private static bool AnyPressThisFrame()
            => Input.anyKeyDown
            || (Input.touchCount > 0 && Input.GetTouch(0).phase == TouchPhase.Began);

        // ── 공개 API ─────────────────────────────────────────────────

        /// <summary>
        /// 이름 입력 흐름을 시작한다(GameFlowManager가 호출). ②까지 완료되면
        /// onNamesEntered(캐릭터명, 오브제명)를 호출하고 ③ 연동 로딩 영상으로 전환한다.
        /// 대기 영상·이름 패널이 모두 미연결이면 입력 단계를 건너뛰고 빈 이름을 보고한 뒤 바로 ③로 간다.
        /// </summary>
        public void BeginNameEntry(Action<string,string> onNamesEntered)
        {
            _onNamesEntered = onNamesEntered;

            // 입력 UI(대기 영상·이름 패널)가 하나도 없으면 입력 단계를 통째로 건너뛴다(빈 이름 → 기존 이름 유지).
            // (대기 영상/이름 패널이 하나라도 있으면 Show()가 띄운 화면에서 사용자 입력을 기다린다.)
            if (!HasAnyEntryUI())
                EnterLoadingPhase();
        }

        private bool HasAnyEntryUI()
            => _attractVideoRoot != null || _characterNamePanel != null || _objectNamePanel != null;

        /// <summary>
        /// 세션·캐릭터 준비 완료 시 호출(GameFlowManager).
        /// Mock: _mockPrepSeconds 경과까지 로딩 영상 유지 후 노출. Server: 즉시 노출.
        /// 입장 버튼이 미연결이면 게이트 없이 즉시 onEnter 실행(폴백).
        /// </summary>
        public void ShowEnterButton(Action onEnter)
        {
            _onEnter = onEnter;

            bool isMock = SessionManager.Instance != null
                       && SessionManager.Instance.dataSourceMode == DataSourceMode.Mock;
            float remaining = isMock
                ? Mathf.Max(0f, _mockPrepSeconds - (Time.unscaledTime - _loadingStartTime))
                : 0f;

            if (_reveal != null) StopCoroutine(_reveal);
            if (remaining > 0f)
                _reveal = StartCoroutine(RevealAfterDelay(remaining));
            else
                RevealReady();
        }

        /// <summary>대기화면 표시(⓪ 대기 영상 → 입력 시 이름). 씬 리로드 재진입 시에도 처음 상태로 되돌린다.</summary>
        public void Show()
        {
            _shownTime = Time.unscaledTime;

            // 모든 단계 요소를 초기 상태로 리셋
            StopAllPhaseCoroutines();
            _attractActive = false;
            SetVideoActive(_attractVideoRoot, _attractCg,     false);
            SetVideoActive(_videoRoot,        _videoCg,       false);
            SetVideoActive(_enterVideoRoot,   _enterVideoCg,  false);
            if (_characterNamePanel != null) _characterNamePanel.SetActive(false);
            if (_objectNamePanel    != null) _objectNamePanel.SetActive(false);
            if (_buttonDelay != null) { StopCoroutine(_buttonDelay); _buttonDelay = null; }
            if (_enterButton != null) _enterButton.gameObject.SetActive(false);

            // ⓪ 대기 영상이 있으면 먼저 띄우고 입력을 기다린다. 없으면 곧장 이름 입력부터.
            if (_attractVideoRoot != null)
            {
                SetVideoActive(_attractVideoRoot, _attractCg, true);   // 대기 영상 재생(Play On Awake·Loop)
                ShowTitle(_attractTitleCg, ref _attractTitleFade, _attractTitleBlink);   // 대기 타이틀(깜빡임/단일)
                _attractActive    = true;
                _attractStartTime = Time.unscaledTime;
            }
            else
            {
                ShowFirstNamePanel();   // 둘 다 없으면 BeginNameEntry가 ③로 보낸다
            }

            if (_root != null) _root.SetActive(true);
            BeginFade(1f, deactivateAtEnd: false);
        }

        /// <summary>대기화면 숨김(최소 표시 시간 보장 후 페이드 아웃·비활성).</summary>
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

        // ── ⓪ 대기 → 이름 전환 ───────────────────────────────────────

        // 대기 영상에서 입력이 들어옴 → 대기 영상 페이드 아웃 후 첫 이름 패널로.
        // 이 시점엔 BeginNameEntry로 콜백이 이미 등록돼 있어, 이름 패널이 없으면 곧장 로딩으로 가도 안전하다.
        private void AdvanceFromAttract()
        {
            _attractActive = false;

            // 대기영상을 떠나는 순간 = 모드 확정 시점. 그때까지 착용 안 됐으면 데스크톱으로 확정한다
            // (착용 중이었으면 그 시점에 이미 VR로 확정돼 있어 이 호출은 무시됨). 매니저 없으면 null-safe no-op.
            XRModeManager.Instance?.ResolveNow();

            // 깜빡임 중지 → 현재 알파에서 멈춘 뒤, 대기 영상 루트 CanvasGroup이 페이드 아웃되며 타이틀도 함께 사라진다.
            if (_attractTitleFade != null) { StopCoroutine(_attractTitleFade); _attractTitleFade = null; }
            _attractFade   = FadeOutVideo(_attractVideoRoot, _attractCg, _attractFadeDuration, _attractFade);

            if (HasNamePanel)
                ShowFirstNamePanel();
            else
                EnterLoadingPhase();
        }

        private bool HasNamePanel => _characterNamePanel != null || _objectNamePanel != null;

        // 첫 이름 패널을 띄운다(캐릭터 우선, 없으면 오브제). 패널이 없으면 아무것도 하지 않는다.
        // (로딩 단계로의 전환은 호출부가 책임진다 — Awake의 Show()에서는 콜백 미등록일 수 있어 여기서 로딩으로 넘기지 않는다.)
        private void ShowFirstNamePanel()
        {
            if (_characterNamePanel != null)
                ShowNamePanel(_characterNamePanel, _charTitleCg, _characterNameInput, ref _charTitleFade);
            else if (_objectNamePanel != null)
                ShowNamePanel(_objectNamePanel, _objTitleCg, _objectNameInput, ref _objTitleFade);
        }

        // ── ①② 이름 입력 단계 ────────────────────────────────────────

        private void OnCharacterNameNext()
        {
            _characterName = _characterNameInput != null ? _characterNameInput.text : "";

            // ② 오브제 단계가 있으면 캐릭터 패널을 즉시 숨기고 오브제 패널로(타이틀 페이드 인).
            //    없으면(마지막 단계) 캐릭터 패널을 켠 채 EnterLoadingPhase가 로딩 영상으로 덮어 전환한다.
            if (_objectNamePanel != null)
            {
                if (_characterNamePanel != null) _characterNamePanel.SetActive(false);
                ShowNamePanel(_objectNamePanel, _objTitleCg, _objectNameInput, ref _objTitleFade);
            }
            else
            {
                EnterLoadingPhase();
            }
        }

        private void OnObjectNameNext()
        {
            _objectName = _objectNameInput != null ? _objectNameInput.text : "";
            EnterLoadingPhase();   // 오브제 패널은 로딩 영상이 페이드 인으로 덮은 뒤 숨긴다(자연 전환)
        }

        // 이름 패널 표시 + 타이틀 Text만 페이드 인 + 입력창 포커스. (모든 이름 단계 공통 — 중복 제거)
        // 패널 본체는 즉시 표시하고, 타이틀 Text(titleCg)만 0→1로 자연스럽게 나타나게 한다.
        private void ShowNamePanel(GameObject panel, CanvasGroup titleCg, TMP_InputField input, ref Coroutine titleFade)
        {
            panel.SetActive(true);
            FadeInTitle(titleCg, ref titleFade);
            input?.ActivateInputField();   // 바로 타이핑·Enter 가능하도록 포커스
        }

        // 타이틀 Text를 0→1 페이드 인한다(미연결이면 즉시). 대기·캐릭터·오브제·시작 4단계 타이틀 공통(중복 제거).
        private void FadeInTitle(CanvasGroup titleCg, ref Coroutine titleFade)
        {
            if (titleCg == null) return;
            if (titleFade != null) StopCoroutine(titleFade);
            if (_titleFadeDuration > 0f)
            {
                titleCg.alpha = 0f;
                titleFade = StartCoroutine(FadeCanvasGroup(titleCg, 1f, _titleFadeDuration));
            }
            else
            {
                titleCg.alpha = 1f;
                titleFade = null;
            }
        }

        // 타이틀 표시: blink면 깜빡임(BlinkTitleRoutine 무한 반복), 아니면 단일 페이드 인.
        // 대기·시작 타이틀 공통(중복 제거). 종료는 StopAllPhaseCoroutines / 단계 전환에서 핸들 중지.
        private void ShowTitle(CanvasGroup titleCg, ref Coroutine titleFade, bool blink)
        {
            if (titleCg == null) return;
            if (blink && _titleFadeDuration > 0f)
            {
                if (titleFade != null) StopCoroutine(titleFade);
                titleCg.alpha = 0f;
                titleFade = StartCoroutine(BlinkTitleRoutine(titleCg));
            }
            else
            {
                FadeInTitle(titleCg, ref titleFade);
            }
        }

        // 타이틀을 페이드 인 → 유지 → 페이드 아웃 → 유지 → 무한 반복(깜빡임). 단계 종료 시 호출부가 중지한다.
        private IEnumerator BlinkTitleRoutine(CanvasGroup cg)
        {
            while (true)
            {
                yield return FadeCanvasGroup(cg, 1f, _titleFadeDuration);
                if (_titleBlinkHold > 0f) yield return new WaitForSecondsRealtime(_titleBlinkHold);
                yield return FadeCanvasGroup(cg, 0f, _titleFadeDuration);
                if (_titleBlinkHold > 0f) yield return new WaitForSecondsRealtime(_titleBlinkHold);
            }
        }

        // ③ 연동 로딩 영상으로 전환하고 이름을 보고한다. 보고 직후 GameFlowManager가 스폰을 시작하며,
        // 스폰 완료 시 ShowEnterButton 으로 ④가 노출된다.
        // 전환은 로딩 영상을 '이름 입력 위로' 페이드 인해 자연스럽게 덮은 뒤 이름 패널을 끈다(스냅 없음).
        // (전제: _videoRoot 가 이름 패널보다 뒤 형제 = 위에 렌더링되어 입력을 덮는다)
        private void EnterLoadingPhase()
        {
            _loadingStartTime = Time.unscaledTime;         // Mock 시뮬 기준 시각

            if (_videoFade != null) StopCoroutine(_videoFade);
            _videoFade = StartCoroutine(FadeInVideoOverPanels());

            // 이름 보고는 영상/타이머를 세팅한 '뒤'에 한다. (Mock은 콜백이 동기로 ShowEnterButton까지
            //  이어지므로, 그 시점에 _loadingStartTime이 이미 설정돼 있어야 한다.)
            var cb = _onNamesEntered;
            _onNamesEntered = null;
            cb?.Invoke(_characterName, _objectName);
        }

        // 로딩 영상을 0→1 페이드 인해 이름 입력 위를 덮은 뒤, 가려진 이름 패널을 끈다(자연 전환).
        private IEnumerator FadeInVideoOverPanels()
        {
            if (_videoRoot != null)
            {
                _videoRoot.SetActive(true);
                if (_videoCg != null && _videoFadeDuration > 0f)
                {
                    _videoCg.alpha = 0f;
                    yield return FadeCanvasGroup(_videoCg, 1f, _videoFadeDuration);
                }
                else if (_videoCg != null)
                {
                    _videoCg.alpha = 1f;
                }
            }

            // 영상이 화면을 덮은 뒤 이름 패널을 끈다(켜진 채 덮였다가 사라져 스냅이 안 보임).
            if (_characterNamePanel != null) _characterNamePanel.SetActive(false);
            if (_objectNamePanel    != null) _objectNamePanel.SetActive(false);
            _videoFade = null;
        }

        // ── ④ 시작 영상 + 입장 ────────────────────────────────────────

        // 준비 완료: 로딩 영상 페이드 아웃 + 시작 영상 페이드 인. 입장 버튼은 _enterButtonDelay 만큼 늦게 노출.
        private void RevealReady()
        {
            _videoFade = FadeOutVideo(_videoRoot, _videoCg, _videoFadeDuration, _videoFade);   // ③ 로딩 영상 사라짐
            FadeInEnterVideo();         // ④ 시작 영상 등장

            if (_enterButton == null)
            {
                OnEnterPressed();       // 버튼 없으면 즉시 진행(폴백)
                return;
            }

            if (_buttonDelay != null) StopCoroutine(_buttonDelay);
            if (_enterButtonDelay > 0f)
                _buttonDelay = StartCoroutine(ShowEnterButtonAfterDelay());
            else
                ActivateEnterButton();
        }

        private IEnumerator ShowEnterButtonAfterDelay()
        {
            yield return new WaitForSecondsRealtime(_enterButtonDelay);
            _buttonDelay = null;
            ActivateEnterButton();
        }

        private void ActivateEnterButton()
        {
            _enterButton.gameObject.SetActive(true);
            _enterButton.interactable = true;
        }

        private IEnumerator RevealAfterDelay(float delay)
        {
            yield return new WaitForSecondsRealtime(delay);
            _reveal = null;
            RevealReady();
        }

        private void OnEnterPressed()
        {
            if (_enterButton != null) _enterButton.interactable = false;   // 중복 클릭 방지
            var cb = _onEnter;
            _onEnter = null;
            cb?.Invoke();
            Hide();
        }

        // ── 영상 페이드 ───────────────────────────────────────────────

        // 영상을 페이드 아웃 후 비활성화(디코딩 정지)한다. 대기 영상·로딩 영상이 공유하는 단일 경로.
        // cg 없거나 duration<=0이면 즉시 끈다. running(진행 중 코루틴)이 있으면 멈춘다.
        // 반환: 새 페이드 코루틴(없으면 null) — 호출부가 자기 핸들에 보관한다.
        private Coroutine FadeOutVideo(GameObject root, CanvasGroup cg, float duration, Coroutine running)
        {
            if (root == null) return null;
            if (running != null) StopCoroutine(running);
            if (cg == null || duration <= 0f)
            {
                root.SetActive(false);
                return null;
            }
            return StartCoroutine(FadeOutAndDisable(root, cg, duration));
        }

        private static IEnumerator FadeOutAndDisable(GameObject root, CanvasGroup cg, float duration)
        {
            yield return FadeCanvasGroup(cg, 0f, duration);
            root.SetActive(false);   // 페이드 끝나면 끔(영상·디코딩 정지)
        }

        // 시작 영상을 활성화하고 페이드 인.
        private void FadeInEnterVideo()
        {
            if (_enterVideoRoot == null) return;
            _enterVideoRoot.SetActive(true);   // 재생 시작(Play On Awake)
            ShowTitle(_enterTitleCg, ref _enterTitleFade, _enterTitleBlink);   // 시작 타이틀(깜빡임/단일)
            if (_enterVideoCg == null || _enterVideoFadeDuration <= 0f)
            {
                if (_enterVideoCg != null) _enterVideoCg.alpha = 1f;
                return;
            }
            if (_enterVideoFade != null) StopCoroutine(_enterVideoFade);
            _enterVideoCg.alpha = 0f;
            _enterVideoFade = StartCoroutine(EnterVideoFadeInRoutine());
        }

        private IEnumerator EnterVideoFadeInRoutine()
        {
            yield return FadeCanvasGroup(_enterVideoCg, 1f, _enterVideoFadeDuration);
            _enterVideoFade = null;
        }

        // ── 루트 페이드 / 공용 ────────────────────────────────────────

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

        private void StopAllPhaseCoroutines()
        {
            if (_reveal         != null) { StopCoroutine(_reveal);         _reveal = null; }
            if (_attractFade    != null) { StopCoroutine(_attractFade);    _attractFade = null; }
            if (_videoFade      != null) { StopCoroutine(_videoFade);      _videoFade = null; }
            if (_enterVideoFade != null) { StopCoroutine(_enterVideoFade); _enterVideoFade = null; }
            if (_attractTitleFade != null) { StopCoroutine(_attractTitleFade); _attractTitleFade = null; }
            if (_charTitleFade  != null) { StopCoroutine(_charTitleFade);  _charTitleFade = null; }
            if (_objTitleFade   != null) { StopCoroutine(_objTitleFade);   _objTitleFade = null; }
            if (_enterTitleFade != null) { StopCoroutine(_enterTitleFade); _enterTitleFade = null; }
        }

        // 영상 오브젝트를 켜고(알파 복구)/끈다. cg 가 있으면 알파도 함께 맞춘다.
        private static void SetVideoActive(GameObject videoRoot, CanvasGroup cg, bool active)
        {
            if (videoRoot == null) return;
            if (cg != null) cg.alpha = active ? 1f : 0f;
            videoRoot.SetActive(active);
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
