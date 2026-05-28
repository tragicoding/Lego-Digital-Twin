using System;
using UnityEngine;
using LegoTwin.Data;
using LegoTwin.Character;
using LegoTwin.Object;
using LegoTwin.UI;

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

        // ── 런타임 참조 (이벤트 해제용) ─────────────────────────────
        private GuideNPCController _currentNPC;
        private string             _currentNpcName;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Start()
        {
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

        // ════════════════════════════════════════════════════════════
        // 세션 로드 완료 → 스폰 + 시나리오 시작
        // ════════════════════════════════════════════════════════════

        private void OnSessionLoaded(SessionData session)
        {
            Debug.Log($"[GameFlowManager] 세션 로드 완료 — {session.character_npc_name} / {session.session_id}");
            _currentNpcName = session.character_npc_name;

            // ① 배치 캐릭터 스폰 (광장 오브제 옆, 모션 씬에서 사용)
            PlacedCharacterController placedCharacter = null;
            if (_characterSpawner != null)
                placedCharacter = _characterSpawner.SpawnPlaced(session);

            // ② 오브제 스폰
            if (_objectSpawner != null && session.assets?.@object != null)
                _objectSpawner.Spawn(session.assets.@object);
            else
                Debug.Log("[GameFlowManager] ObjectSpawner 없음 또는 오브제 데이터 없음 — 오브제 스폰 생략");

            // ③ 가이드 NPC 스폰
            if (_characterSpawner == null)
            {
                Debug.LogError("[GameFlowManager] GeneratedCharacterSpawner가 연결되지 않았습니다. " +
                               "Inspector에서 _characterSpawner 필드를 연결하세요.");
                return;
            }

            var npc = _characterSpawner.SpawnGuide(session);
            if (npc == null)
            {
                Debug.LogError("[GameFlowManager] 가이드 NPC 스폰 실패. " +
                               "mockCharacterPrefab 또는 guideSpawnPoint 를 확인하세요.");
                return;
            }

            // ④ 배치 캐릭터 → NPC에 연결 (모션 씬 Step 3 에서 사용)
            npc.placedCharacter = placedCharacter;

            // ⑤ NPC 이벤트 구독
            _currentNPC = npc;
            npc.OnDialogueChanged           += OnDialogueChanged;
            npc.OnFreeModeSwitched          += OnFreeModeSwitched;
            npc.OnGuideFinished             += OnGuideFinished;
            npc.OnMotionPromptRequested     += OnMotionPromptRequested;
            npc.OnPlayerTeleportRequested   += OnPlayerTeleportRequested;

            // ⑥ 시나리오 시작 ── GuideScenarioRoutine() 코루틴 실행
            npc.StartGuideScenario();
        }

        // ════════════════════════════════════════════════════════════
        // NPC 이벤트 핸들러
        // ════════════════════════════════════════════════════════════

        private void OnDialogueChanged(string text)
        {
            Debug.Log($"[말풍선] {text}");
            _dialogueUI?.Show(_currentNpcName, text);
        }

        /// <summary>
        /// [Step 5] 자유 모드 팝업 표시.
        /// "자유모드로 전환되었습니다." 팝업을 _freeModePopup에 연결하면 자동 활성화된다.
        /// </summary>
        private void OnFreeModeSwitched()
        {
            Debug.Log("[GameFlowManager] 자유 모드 전환 팝업 표시");
            _freeModePopup?.Show();   // 페이드 인 → displayDuration 대기 → 페이드 아웃 자동 실행
        }

        /// <summary>
        /// [Step 5] 가이드 시나리오 완료 → 자유 모드 전환.
        ///
        /// TODO: PlazaManager 구현 후 아래 주석 해제:
        ///   LegoTwin.Plaza.PlazaManager.Instance.EnterPlaza();
        /// </summary>
        private void OnGuideFinished()
        {
            Debug.Log("[GameFlowManager] 가이드 종료 → 자유 모드 전환");
            _dialogueUI?.Hide();
            UnsubscribeNPCEvents();

            // TODO: LegoTwin.Plaza.PlazaManager.Instance.EnterPlaza();
        }

        /// <summary>
        /// [Step 3] 모션 프롬프트 입력 요청.
        /// VR 키보드 또는 입력 UI를 열고, 입력 완료 시 callback(text) 를 호출한다.
        ///
        /// TODO: VR 입력 UI 연동 후 아래 더미 코드 교체:
        ///   _inputUI.Open(inputText => callback(inputText));
        /// </summary>
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
            Debug.Log($"[GameFlowManager] 플레이어 순간이동 → {destination}");
            _playerFollowGuide.transform.position = destination;
        }

        private void OnMotionPromptRequested(Action<string> callback)
        {
            Debug.Log("[GameFlowManager] 모션 입력 요청 (임시: '춤춰줘' 즉시 반환)");

            // ── 임시 더미 ─────────────────────────────────────────────
            callback?.Invoke("춤춰줘");
            // ──────────────────────────────────────────────────────────
        }

        // ════════════════════════════════════════════════════════════
        // 내부 유틸
        // ════════════════════════════════════════════════════════════

        private void UnsubscribeNPCEvents()
        {
            if (_currentNPC == null) return;
            _currentNPC.OnDialogueChanged           -= OnDialogueChanged;
            _currentNPC.OnFreeModeSwitched          -= OnFreeModeSwitched;
            _currentNPC.OnGuideFinished             -= OnGuideFinished;
            _currentNPC.OnMotionPromptRequested     -= OnMotionPromptRequested;
            _currentNPC.OnPlayerTeleportRequested   -= OnPlayerTeleportRequested;
            _currentNPC = null;
        }
    }
}
