using System;
using UnityEngine;
using LegoTwin.Core;
using LegoTwin.Data;
using LegoTwin.Character;
using LegoTwin.Object;
using LegoTwin.UI;
using LegoTwin.Network;

namespace LegoTwin.Managers
{
    /// <summary>
    /// 가이드 모드 전체 흐름 관리 진입점.
    ///
    /// 역할:
    ///   1. SessionManager.OnSessionLoaded 구독
    ///   2. 캐릭터(가이드 + 배치) · 오브제 스폰 지시
    ///   3. GuideNPCController 이벤트 구독 (말풍선, 모션 입력, 자유 모드 전환)
    ///   4. StartGuideScenario() 호출 → 등장 씬부터 자동 진행
    ///
    /// Mock → Server 전환 시 이 파일은 수정하지 않는다.
    /// SessionManager.dataSourceMode 를 Server 로 바꾸면 자동 전환된다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] 씬에 이 컴포넌트를 가진 GameFlowManager GameObject 배치
    ///   [ ] _characterSpawner 필드에 GeneratedCharacterSpawner 연결
    ///   [ ] _objectSpawner    필드에 GeneratedObjectSpawner    연결 (없으면 스킵)
    ///   [ ] _freeModePopup    필드에 자유모드 팝업 GameObject   연결
    /// </summary>
    public class GameFlowManager : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("스포너")]
        [Tooltip("씬의 GeneratedCharacterSpawner 연결")]
        [SerializeField] private GeneratedCharacterSpawner _characterSpawner;

        [Tooltip("씬의 GeneratedObjectSpawner 연결 (없으면 오브제 스폰 생략)")]
        [SerializeField] private GeneratedObjectSpawner _objectSpawner;

        [Header("플레이어")]
        [Tooltip("XR Origin(또는 Camera Rig 루트)에 붙은 PlayerFollowGuide 연결")]
        [SerializeField] private PlayerFollowGuide _playerFollowGuide;

        [Header("UI")]
        [Tooltip("말풍선 UI — DialogueUI 컴포넌트가 붙은 Canvas GameObject 연결")]
        [SerializeField] private DialogueUI _dialogueUI;

        [Tooltip("자유 모드 전환 시 표시할 팝업 — FreeModePopupUI 컴포넌트가 붙은 GameObject 연결")]
        [SerializeField] private FreeModePopupUI _freeModePopup;

        [Tooltip("모션 프롬프트 입력 UI — MotionPromptUI 컴포넌트가 붙은 GameObject 연결")]
        [SerializeField] private MotionPromptUI _motionPromptUI;

        [Tooltip("시그니처 동작 설정 확인 UI — SignatureMotionConfirmUI 컴포넌트가 붙은 GameObject 연결")]
        [SerializeField] private SignatureMotionConfirmUI _signatureConfirmUI;

        [Tooltip("지도 텔레포트 UI GameObject. 시작 시 자동 비활성화 → 자유 모드 진입 시 활성화")]
        [SerializeField] private GameObject _mapTeleportUI;

        [Tooltip("종료 UI GameObject. 시작 시 자동 비활성화 → 자유 모드 진입 시 활성화")]
        [SerializeField] private GameObject _quitUI;

        [Header("디버그")]
        [Tooltip("체크 시 지정 키로 가이드 모드를 즉시 스킵. 전시 빌드에서는 해제 권장.")]
        [SerializeField] private bool _enableGuideSkip = true;

        [Tooltip("가이드 모드 스킵 디버그 키 (노트북 F키는 Fn 가로채기 주의 — 기본 Backspace)")]
        [SerializeField] private KeyCode _skipGuideKey = KeyCode.Backspace;

        // ── 런타임 참조 (이벤트 해제용) ─────────────────────────────
        private GuideNPCController       _currentNPC;
        private PlacedCharacterController _placedCharacter;
        private string                   _currentNpcName;

        /// <summary>가이드 모드 진행 중이면 true. FreeCameraController 등에서 이동 제한에 사용.</summary>
        public static bool IsGuideMode { get; private set; } = true;

        // ── 스폰된 오브젝트 참조 (PlazaManager에 전달용) ─────────────
        private GameObject _spawnedCharacterGO;
        private GameObject _spawnedObjectGO;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Start()
        {
            // 자유 모드 진입 전까지 지도/종료 UI 숨김 — 인스펙터에서 끄지 않아도 자동 비활성화.
            // (OnFreeModeSwitched 에서 다시 활성화됨)
            if (_mapTeleportUI != null) _mapTeleportUI.SetActive(false);
            if (_quitUI != null)        _quitUI.SetActive(false);

            if (SessionManager.Instance == null)
            {
                Debug.LogError("[GameFlowManager] SessionManager 인스턴스를 찾을 수 없습니다. " +
                               "씬에 SessionManager GameObject가 있는지 확인하세요.");
                return;
            }

            // 씬 재진입 등으로 이미 로드된 세션이 있으면 즉시 처리
            if (SessionManager.Instance.CurrentSession != null)
            {
                OnSessionLoaded(SessionManager.Instance.CurrentSession);
                return;
            }

            // 세션 로드 완료 이벤트 구독 → SessionManager.Start() 에서 자동 로드됨
            SessionManager.Instance.OnSessionLoaded += OnSessionLoaded;
        }

        private void OnDestroy()
        {
            if (SessionManager.Instance != null)
                SessionManager.Instance.OnSessionLoaded -= OnSessionLoaded;

            UnsubscribeNPCEvents();
        }

        private void Update()
        {
            // (디버그) 가이드 모드 스킵 — 시나리오 중단 후 즉시 자유 모드 전환
            // IsGuideMode(static) 대신 _currentNPC 존재로 판단 → 도메인 리로드 설정 영향 없음
            if (!_enableGuideSkip || _currentNPC == null) return;

            KeyCode key = _skipGuideKey == KeyCode.None ? KeyCode.Backspace : _skipGuideKey;
            if (Input.GetKeyDown(key))
                SkipGuide();
        }

        /// <summary>(디버그) 진행 중인 가이드 시나리오를 중단하고 자유 모드로 즉시 전환.</summary>
        private void SkipGuide()
        {
            _currentNPC.AbortScenario();

            // 정상 흐름 Step 2와 동일하게 창작물 앞으로 순간이동 (이벤트 구독 중인 지금 호출)
            _currentNPC.JumpToCreationView();

            // 진행 중 열려 있을 수 있는 입력/확인 UI 정리 (콜백 실행 없이 닫음)
            _motionPromptUI?.Close();
            _signatureConfirmUI?.Close();

            OnFreeModeSwitched();  // 자유모드 전환 팝업 표시 (정상 흐름 Step 5와 동일)
            OnGuideFinished();     // IsGuideMode=false · 말풍선 숨김 · Plaza 진입 처리
        }

        // ════════════════════════════════════════════════════════════
        // 세션 로드 완료 → 스폰 + 시나리오 시작
        // ════════════════════════════════════════════════════════════

        // 스폰은 콜백 기반(Mock=동기, Server=TriLib 비동기). 어느 모드든 동일하게 동작한다.
        private void OnSessionLoaded(SessionData session)
        {
            _currentNpcName = session.character_npc_name;

            if (_characterSpawner == null)
            {
                Debug.LogError("[GameFlowManager] GeneratedCharacterSpawner가 연결되지 않았습니다. " +
                               "Inspector에서 _characterSpawner 필드를 연결하세요.");
                return;
            }

            // ① 오브제 스폰 (Server 모드는 비동기 → 완료 참조는 OnGuideFinished 시점에 회수)
            if (_objectSpawner != null && session.assets?.@object != null)
            {
                _objectSpawner.Spawn(session.assets.@object);
                _spawnedObjectGO = _objectSpawner.GetSpawnedObject();
            }

            // ② 배치 캐릭터 스폰 — 준비되면 참조 저장(+가이드가 이미 있으면 늦은 연결 보정)
            _characterSpawner.SpawnPlaced(session, placed =>
            {
                _placedCharacter    = placed;
                _spawnedCharacterGO = placed?.gameObject;
                if (_currentNPC != null) _currentNPC.placedCharacter = placed;
            });

            // ③ 가이드 NPC 스폰 — 준비되면 연결·구독·시나리오 시작
            _characterSpawner.SpawnGuide(session, npc =>
            {
                if (npc == null)
                {
                    Debug.LogError("[GameFlowManager] 가이드 NPC 스폰 실패. " +
                                   "mockCharacterPrefab 또는 guideSpawnPoint 를 확인하세요.");
                    return;
                }

                // 배치 캐릭터 연결 (Mock=이미 set, Server=②의 늦은 연결이 보완)
                npc.placedCharacter = _placedCharacter;

                // 말풍선 UI가 NPC 머리 위를 따라다니도록 타겟 주입
                _dialogueUI?.SetFollowTarget(npc.transform);

                _currentNPC = npc;
                SubscribeNPCEvents(npc);

                // 시나리오 시작 ── GuideScenarioRoutine() 코루틴 실행
                npc.StartGuideScenario();
            });
        }

        // ════════════════════════════════════════════════════════════
        // NPC 이벤트 핸들러
        // ════════════════════════════════════════════════════════════

        private void OnDialogueChanged(string text)
        {
            _dialogueUI?.Show(_currentNpcName, text);
        }

        /// <summary>
        /// [Step 5] 자유 모드 팝업 표시.
        /// "자유모드로 전환되었습니다." 팝업을 _freeModePopup에 연결하면 자동 활성화된다.
        /// </summary>
        private void OnFreeModeSwitched()
        {
            _freeModePopup?.Show();   // 페이드 인 → displayDuration 대기 → 페이드 아웃 자동 실행

            // 자유 모드 진입 시 지도/종료 UI 활성화 (가이드 중에는 비활성으로 시작)
            if (_mapTeleportUI != null) _mapTeleportUI.SetActive(true);
            if (_quitUI != null)        _quitUI.SetActive(true);
        }

        /// <summary>
        /// [Step 5] 가이드 시나리오 완료 → 자유 모드 전환.
        ///
        /// TODO: PlazaManager 구현 후 아래 주석 해제:
        ///   LegoTwin.Plaza.PlazaManager.Instance.EnterPlaza();
        /// </summary>
        private void OnGuideFinished()
        {
            IsGuideMode = false;
            _dialogueUI?.Hide();
            UnsubscribeNPCEvents();

            var plaza = LegoTwin.Plaza.PlazaManager.Instance;
            if (plaza == null)
            {
                Debug.LogWarning("[GameFlowManager] PlazaManager.Instance 없음 — 씬에 PlazaManager GameObject가 있는지 확인하세요.");
                return;
            }

            // Server Mode에서 GLB 로드는 비동기 → Spawn() 직후 null이므로
            // 가이드 시나리오가 끝난 이 시점(~20초 후)에 완료된 참조를 가져온다
            var objectGO = _spawnedObjectGO ?? _objectSpawner?.GetSpawnedObject();

            plaza.RegisterCurrentSession(
                SessionManager.Instance.CurrentSession,
                _spawnedCharacterGO,
                objectGO);

            plaza.EnterPlaza();

            // 스킵 등으로 오브제 GLB 로드가 아직 안 끝난 경우(Server Mode):
            // 로드 완료까지 기다렸다가 현재 창작물 뷰를 늦게 부착한다.
            bool isServer = SessionManager.Instance != null
                         && SessionManager.Instance.dataSourceMode == DataSourceMode.Server;
            if (objectGO == null && isServer && _objectSpawner != null)
                StartCoroutine(AttachCurrentCreationWhenObjectLoaded(plaza));
        }

        /// <summary>
        /// (스킵 보강) 오브제 GLB 비동기 로드가 끝날 때까지 대기한 뒤
        /// 현재 관람객 창작물 뷰를 광장에 부착한다. (이미 부착됐으면 PlazaManager가 무시)
        /// </summary>
        private System.Collections.IEnumerator AttachCurrentCreationWhenObjectLoaded(
            LegoTwin.Plaza.PlazaManager plaza)
        {
            const float timeout = 30f;
            float elapsed = 0f;
            GameObject obj = null;

            while (elapsed < timeout)
            {
                obj = _objectSpawner.GetSpawnedObject();
                if (obj != null) break;
                elapsed += Time.deltaTime;
                yield return null;
            }

            if (obj == null)
            {
                Debug.LogWarning("[GameFlowManager] 오브제 로드 대기 타임아웃 — 현재 창작물 뷰 부착 생략");
                yield break;
            }

            plaza.AttachCurrentObjectLate(obj);
        }

        /// <summary>
        /// [Step 2] 창작물 위치로 플레이어 순간이동.
        /// _playerFollowGuide 가 XR Origin 루트에 붙어있으므로 그 transform 을 직접 이동한다.
        /// </summary>
        private void OnPlayerTeleportRequested(Vector3 destination)
        {
            if (_playerFollowGuide == null)
            {
                Debug.LogWarning("[GameFlowManager] _playerFollowGuide 미연결 — 플레이어 순간이동 생략");
                return;
            }
            // CC-safe 텔레포트는 공용 유틸에 위임 (지도 UI 등과 동일 로직 공유).
            PlayerTeleporter.Teleport(_playerFollowGuide.transform, destination);
        }

        /// <summary>
        /// [Step 3] 모션 프롬프트 입력 요청.
        /// MotionPromptUI 입력창을 열고, 플레이어가 확인을 누르면 callback(text) 가 호출된다.
        /// </summary>
        private void OnMotionPromptRequested(Action<string> callback)
        {
            if (_motionPromptUI == null)
            {
                Debug.LogWarning("[GameFlowManager] _motionPromptUI 미연결 — Inspector에서 연결하세요. 임시로 '춤춰줘' 반환.");
                callback?.Invoke("춤춰줘");
                return;
            }

            _motionPromptUI.Show(callback);
        }

        /// <summary>
        /// [Step 3-1] 시그니처 동작 설정 확인 다이얼로그 표시.
        /// 예 선택 시 SessionData에 저장 + Server Mode이면 API 전송.
        /// </summary>
        // onResult: 예→true, 아니오→false — GuideNPCController가 루프 여부를 판단
        private void OnSignatureMotionConfirmRequested(string motionInput, Action<bool> onResult)
        {
            if (_signatureConfirmUI == null)
            {
                Debug.LogWarning("[GameFlowManager] _signatureConfirmUI 미연결 — 시그니처 설정 생략");
                onResult?.Invoke(false);
                return;
            }

            // 안내 모드: 나가기 버튼 없음 (showExitButton = false)
            _signatureConfirmUI.Show(
                confirmed =>
                {
                    if (confirmed) ApplySignatureMotion(motionInput);
                    onResult?.Invoke(confirmed);
                },
                showExitButton: false
            );
        }

        private void ApplySignatureMotion(string motionInput)
        {
            // 관람객이 방금 본 "특정 클립"을 그대로 저장. 클립을 못 읽으면 MotionType 이름으로 폴백.
            var clipName = _placedCharacter != null ? _placedCharacter.LastPromptClipName : null;
            if (string.IsNullOrEmpty(clipName))
                clipName = MotionPromptParser.Parse(motionInput).ToString();

            SessionManager.Instance?.SetSignatureMotion(clipName);
        }

        // ════════════════════════════════════════════════════════════
        // 내부 유틸
        // ════════════════════════════════════════════════════════════

        private void SubscribeNPCEvents(GuideNPCController npc)
        {
            npc.OnDialogueChanged                   += OnDialogueChanged;
            npc.OnFreeModeSwitched                  += OnFreeModeSwitched;
            npc.OnGuideFinished                     += OnGuideFinished;
            npc.OnMotionPromptRequested             += OnMotionPromptRequested;
            npc.OnPlayerTeleportRequested           += OnPlayerTeleportRequested;
            npc.OnSignatureMotionConfirmRequested   += OnSignatureMotionConfirmRequested;
        }

        private void UnsubscribeNPCEvents()
        {
            if (_currentNPC == null) return;
            _currentNPC.OnDialogueChanged                   -= OnDialogueChanged;
            _currentNPC.OnFreeModeSwitched                  -= OnFreeModeSwitched;
            _currentNPC.OnGuideFinished                     -= OnGuideFinished;
            _currentNPC.OnMotionPromptRequested             -= OnMotionPromptRequested;
            _currentNPC.OnPlayerTeleportRequested           -= OnPlayerTeleportRequested;
            _currentNPC.OnSignatureMotionConfirmRequested   -= OnSignatureMotionConfirmRequested;
            _currentNPC = null;
        }
    }
}
