using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.AI;
using LegoTwin.Character;
using LegoTwin.Core;
using LegoTwin.Data;
using LegoTwin.Managers;
using LegoTwin.Network;
using LegoTwin.Object;
using LegoTwin.UI;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// 광장 관리자 — 자유 모드 진입 시 이전 관람객들의 창작물을 배치.
    ///
    /// 동작 흐름:
    ///   1. EnterPlaza() 호출
    ///   2. Mock: mock_plaza.json 로드 / Server: GET /unity/plaza/sessions
    ///   3. 각 세션의 캐릭터 + 오브제를 spawnPoints[i] 에 배치
    ///   4. 좋아요 1위 오브제에 별(star) 표시
    ///   5. WebSocketManager.OnLikesUpdated 구독 → 실시간 좋아요 갱신
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] spawnPoints 배열에 광장 배치 위치들 연결 (캐릭터+오브제 쌍)
    ///   [ ] mockCharacterPrefab, mockObjectPrefab 연결 (Mock Mode용)
    ///   [ ] PlazaSessionView 프리팹 연결
    ///   [ ] 좋아요 UI (별 표시) 구현 후 PlazaSessionView.SetTopLiked() 에 연결
    /// </summary>
    public class PlazaManager : MonoBehaviour
    {
        public static PlazaManager Instance { get; private set; }

        private const int MaxPlazaSessions = 31;

        [Header("배치 위치 (캐릭터+오브제 쌍, 순서대로)")]
        public Transform[] spawnPoints;

        [Header("Mock Mode Prefab (복수 등록 → 세션 순서대로 순환 배정)")]
        public GameObject[] mockCharacterPrefabs;
        public GameObject[] mockObjectPrefabs;

        [Header("Server Mode — 사전 제작 캐릭터 Resources 경로")]
        [Tooltip("Assets/**/Resources 아래 경로. 예: Resources/PrebuiltCharacters/Character_1.fbx")]
        public string prebuiltCharacterResourcePrefix = "PrebuiltCharacters/Character_";

        [Header("스폰 설정")]
        public float characterSpawnScale    = 6f;
        public float objectSpawnScale       = 12f;
        public float characterForwardOffset = 3f;   // 캐릭터: spawnPoint 앞쪽
        public float objectBackOffset       = 3f;   // 오브제:  spawnPoint 뒤쪽
        public float viewHeightOffset       = 5f;   // 투표 UI: spawnPoint에서 위로 띄울 높이
        [Tooltip("오브제 위 말풍선(BubblePanel) 높이 오프셋 (m, 오브제 바운즈 상단 기준)")]
        public float bubbleObjectHeightOffset = 1f;  // 말풍선: 오브제 머리 위로 띄울 높이
        [Tooltip("오브제 위 말풍선(패널) 크기 배율 (1=원본, 키우려면 1보다 크게)")]
        public float bubbleObjectScale = 2f;         // 말풍선 패널 크기 배율
        [Tooltip("말풍선 글씨 크기 배율 (패널 크기와 독립. 1=기본, 글씨만 키우거나 줄임)")]
        public float bubbleTextScale = 1f;           // 말풍선 글씨 크기 배율(패널과 독립)
        [Tooltip("말풍선 좌우 오프셋 (m, 카메라 기준. +면 화면 오른쪽으로)")]
        public float bubbleObjectRightOffset = 0f;   // 말풍선 좌우 위치
        [Tooltip("좋아요 패널 높이 오프셋 (m, 플레이어 눈높이 기준. 0=눈높이, +면 위로)")]
        public float likeObjectHeightOffset = 0f;    // 좋아요 패널: 플레이어 눈높이에서 위로 띄울 높이
        [Tooltip("좋아요 패널 크기 배율 (1=원본, 키우려면 1보다 크게)")]
        public float likeObjectScale = 2f;           // 좋아요 패널 크기 배율
        [Tooltip("좋아요 패널 좌우 오프셋 (m, 카메라 기준. +면 화면 오른쪽으로)")]
        public float likeObjectRightOffset = 0f;     // 좋아요 패널 좌우 위치
        [Tooltip("좋아요 패널을 오브제 앞으로 당길 추가 여유 (m, 크기 비례분에 더함). 0이면 앞면에 딱 붙음")]
        public float likeObjectForwardOffset = 0.3f; // 좋아요 패널: 앞면에서 추가로 띄울 고정 여유
        [Tooltip("오브제 크기 자동 조정: 오브제 반깊이 × 이 값만큼 앞으로. 1=앞면, <1이면 안쪽(큰 오브제일수록 가까이). 0이면 고정값만")]
        public float likeObjectForwardSizeFactor = 0.65f; // 오브제 크기 비례 전진(낮출수록 큰 오브제가 가까이)

        [Header("이전 창작물 캐릭터 배회 (NavMesh — 광장 바닥에 NavMesh 베이크 필요)")]
        [Tooltip("배회 중심 Transform. 비우면(권장) 각 캐릭터가 '자기 스폰 자리'를 중심으로 배회해 " +
                 "자연히 분산된다. 지정하면 모든 캐릭터가 그 한 점을 공유(전역 공용 중심).")]
        public Transform wanderCenter;
        [Tooltip("배회 반경 (미터) — 중심에서 이 반경 안의 NavMesh 위를 무작위로 돌아다님")]
        public float wanderRadius   = 15f;
        [Tooltip("배회 이동 속도 (m/s)")]
        public float wanderMoveSpeed = 2.5f;
        [Tooltip("플레이어가 이 거리 안에 들어오면 멈춰서 플레이어를 보고 시그니처 동작 (미터)")]
        public float approachRadius  = 3.5f;
        [Tooltip("목적지 도착 후 최소/최대 대기 시간 (초)")]
        public float wanderWaitMin   = 1f;
        public float wanderWaitMax   = 3f;

        [Header("PlazaSessionView 프리팹 (UI + 좋아요 포함)")]
        public PlazaSessionView sessionViewPrefab;

        [Header("시그니처 동작 — 광장 캐릭터 컴포넌트 자동 주입")]
        [Tooltip("NPC_AnimatorController 연결 — 모션 Override에 필요")]
        public RuntimeAnimatorController characterAnimatorController;
        [Tooltip("NPC_MotionLibrary 연결 — 시그니처 동작 클립 조회에 필요")]
        public MixamoMotionLibrary motionLibrary;

        [Header("내 창작물 인터랙션 (자유 모드)")]
        [Tooltip("동작 입력 UI — MotionPromptUI 컴포넌트가 붙은 GameObject 연결")]
        public MotionPromptUI motionPromptUI;
        [Tooltip("시그니처 확인 다이얼로그 — SignatureMotionConfirmUI 컴포넌트가 붙은 GameObject 연결")]
        public SignatureMotionConfirmUI signatureConfirmUI;
        [Tooltip("내 창작물 접근 시 표시할 인터랙션 패널 프리팹 ('동작 입력하기' 버튼 포함)")]
        public GameObject ownCreationInteractPanelPrefab;

        [Header("동작 입력 패널 위치 (좋아요 패널과 독립)")]
        [Tooltip("눈높이 기준 높이 오프셋 (m). 음수면 눈높이보다 '아래'로 내려간다(이 패널은 눈높이를 낮추는 용도). " +
                 "효과가 약하면 더 큰 음수로(캐릭터가 크게 스케일된 광장에선 -1~-3 정도 필요할 수 있음).")]
        public float interactPanelHeightOffset = -0.5f;
        [Tooltip("좌우 오프셋 (m, 카메라 기준. +면 화면 오른쪽으로)")]
        public float interactPanelRightOffset = 0f;
        [Tooltip("캐릭터 앞으로 당길 추가 여유 (m, 크기 비례분에 더함). 0이면 앞면에 딱 붙음")]
        public float interactPanelForwardOffset = 0.3f;
        [Tooltip("크기 자동 조정: 캐릭터 반깊이 × 이 값만큼 앞으로. 1=앞면, <1이면 안쪽(큰 대상일수록 가까이). 0이면 고정값만")]
        public float interactPanelForwardSizeFactor = 0.65f;
        [Tooltip("패널 크기 배율 (1=프리팹 원본 크기. 키우려면 1보다 크게)")]
        public float interactPanelScale = 1f;

        private readonly List<PlazaSessionView> _views = new();
        private readonly Dictionary<string, int> _likeCounts = new();
        private string _topSessionId;
        private GameObject _lastSpawnedObjectGO;   // 직전 세션에서 스폰된 오브제(말풍선 배치용)

        // 현재 관람객 창작물 (가이드 모드에서 스폰된 오브젝트)
        private SessionData _currentSession;
        private GameObject  _currentCharacterGO;
        private GameObject  _currentObjectGO;
        private bool        _currentViewAttached;   // 현재 창작물 뷰 부착 완료 여부 (중복 방지)

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;

            // 새 세션(씬 리로드)마다 이전 관람객의 좋아요 중복 방지 기록을 초기화.
            // static은 씬 리로드로 자동 초기화되지 않으므로 명시적으로 비운다.
            LikeSystem.ResetVotes();
        }

        private void OnDestroy()
        {
            // 씬 리로드(A안 종료 흐름)로 파괴될 때, DontDestroyOnLoad인 WebSocketManager의
            // OnLikesUpdated에 죽은 핸들러가 남지 않도록 구독을 해제한다.
            // (ExitPlaza는 정상 종료 시에만 호출되므로 여기서 한 번 더 보장. -=는 idempotent)
            if (Instance == this && WebSocketManager.Instance != null)
                WebSocketManager.Instance.OnLikesUpdated -= HandleLikesUpdated;
        }

        // ════════════════════════════════════════════════════════════
        // 진입점 — 가이드 모드 완료 후 호출
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// GameFlowManager가 OnGuideFinished 직전에 호출.
        /// 이미 씬에 스폰된 현재 관람객 창작물 GO를 전달해 투표 시스템 부착에 사용.
        /// </summary>
        public void RegisterCurrentSession(SessionData session, GameObject charGo, GameObject objGo)
        {
            _currentSession      = session;
            _currentCharacterGO  = charGo;
            _currentObjectGO     = objGo;
            _currentViewAttached = false;
        }

        /// <summary>
        /// (스킵 보강) 비동기 로드로 늦게 도착한 현재 관람객 오브제를 받아 뷰를 부착한다.
        /// EnterPlaza 시점에 오브제가 아직 null이라 부착을 못 한 경우 GameFlowManager가 호출.
        /// 이미 부착됐거나 오브제가 없으면 무시.
        /// </summary>
        public void AttachCurrentObjectLate(GameObject objGo)
        {
            if (_currentViewAttached || objGo == null) return;
            _currentObjectGO = objGo;
            AttachViewToCurrentSession();
        }

        /// <summary>자유 모드 진입 시 호출. 광장 데이터 로드 + 배치 시작.</summary>
        public void EnterPlaza()
        {
            StartCoroutine(LoadAndSpawnPlaza());

            if (WebSocketManager.Instance != null)
            {
                // Server Mode: WebSocket 연결을 시작해 likes_updated 이벤트 수신
                if (SessionManager.Instance?.dataSourceMode == DataSourceMode.Server)
                    WebSocketManager.Instance.StartListening();

                WebSocketManager.Instance.OnLikesUpdated += HandleLikesUpdated;
            }
        }

        public void ExitPlaza()
        {
            if (WebSocketManager.Instance != null)
                WebSocketManager.Instance.OnLikesUpdated -= HandleLikesUpdated;

            foreach (var v in _views)
                if (v != null) Destroy(v.gameObject);
            _views.Clear();
        }

        // Mock 모드에서 앱 종료 시 현재 세션을 mock_plaza.json에 자동 저장
        private void OnApplicationQuit()
        {
            if (SessionManager.Instance?.dataSourceMode != DataSourceMode.Mock) return;
            SaveCurrentSessionToMockPlaza();
        }

        // ════════════════════════════════════════════════════════════
        // Mock 모드 — 현재 세션 저장 (다음 관람객에게 이전 창작물로 표시)
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 현재 관람객 세션을 mock_plaza.json에 추가한다.
        /// Mock 모드 전용. 앱 종료 시 자동 호출되며 수동 호출도 가능.
        /// </summary>
        public void SaveCurrentSessionToMockPlaza()
        {
            var session = _currentSession ?? SessionManager.Instance?.CurrentSession;
            if (session == null)
            {
                Debug.LogWarning("[PlazaManager] 저장할 현재 세션 없음");
                return;
            }

            var plaza = MockPlazaStore.Load() ?? new PlazaResponse();
            if (plaza.sessions == null) plaza.sessions = new List<PlazaSessionData>();

            // 동일 원본 session_id 중복 방지
            if (plaza.sessions.Any(s => s.session_id == session.session_id))
            {
                return;
            }

            // 새 plaza_XXX ID 생성 (기존 최대값 + 1)
            int maxNum = plaza.sessions
                .Where(s => s.session_id.StartsWith("plaza_"))
                .Select(s => int.TryParse(s.session_id.Substring(6), out int n) ? n : 0)
                .DefaultIfEmpty(0)
                .Max();
            string newId = $"plaza_{(maxNum + 1):D3}";

            // 플레이 중 받은 좋아요 수 반영
            _likeCounts.TryGetValue(session.session_id, out int currentLikes);

            var newSession = new PlazaSessionData
            {
                session_id         = newId,
                character_npc_name = session.character_npc_name,
                bubble_text        = session.bubble_text        ?? "",
                signature_motion   = session.signature_motion   ?? "",
                character_number   = session.character_number,
                likes              = currentLikes,
                is_top_liked       = false,
                assets             = session.assets
            };
            plaza.sessions.Add(newSession);

            // 최대 세션 수 초과 시 — 오래된 순으로 제거 (likes 상위 8위는 보호)
            if (plaza.sessions.Count > MaxPlazaSessions)
            {
                // likes 상위 8개 세션은 제거 대상에서 제외
                var protectedIds = plaza.sessions
                    .OrderByDescending(s => s.likes)
                    .Take(8)
                    .Select(s => s.session_id)
                    .ToHashSet();

                // 보호 대상 제외 후 가장 오래된 세션 선택
                var toRemove = plaza.sessions
                    .Where(s => !protectedIds.Contains(s.session_id))
                    .OrderBy(s => PlazaSessionNumber(s.session_id))
                    .FirstOrDefault();

                // 모두 상위 8위 이내인 극단적 경우 — 보호 없이 가장 오래된 것 제거
                if (toRemove == null)
                    toRemove = plaza.sessions
                        .OrderBy(s => PlazaSessionNumber(s.session_id))
                        .First();

                plaza.sessions.Remove(toRemove);
            }

            // top_session_id 재계산
            plaza.top_session_id = plaza.sessions
                .OrderByDescending(s => s.likes)
                .First().session_id;

            MockPlazaStore.Save(plaza);
        }

        // plaza_XXX 형식에서 숫자 추출 (정렬용). 형식 불일치 시 int.MaxValue 반환.
        private static int PlazaSessionNumber(string sessionId)
        {
            if (sessionId.StartsWith("plaza_") &&
                int.TryParse(sessionId.Substring(6), out int n)) return n;
            return int.MaxValue;
        }

        // ════════════════════════════════════════════════════════════
        // 로드 + 배치
        // ════════════════════════════════════════════════════════════

        private IEnumerator LoadAndSpawnPlaza()
        {
            PlazaResponse plaza = null;

            if (SessionManager.Instance?.dataSourceMode == Managers.DataSourceMode.Mock)
            {
                plaza = MockPlazaStore.Load();
            }
            else
            {
                yield return ApiClient.Instance.FetchPlazaSessions(r => plaza = r);
            }

            if (plaza == null || plaza.sessions == null)
            {
                Debug.LogWarning("[PlazaManager] 광장 데이터 없음");
                yield break;
            }

            _topSessionId = plaza.top_session_id;

            // 현재 관람객 창작물에 투표 뷰 부착
            AttachViewToCurrentSession();

            for (int i = 0; i < plaza.sessions.Count; i++)
            {
                if (i >= spawnPoints.Length)
                {
                    Debug.LogWarning($"[PlazaManager] spawnPoints 부족 ({spawnPoints.Length}개) — Inspector에 더 추가하세요. (필요: {plaza.sessions.Count}개)");
                    break;
                }

                var session = plaza.sessions[i];
                var point   = spawnPoints[i];

                _likeCounts[session.session_id] = session.likes;

                // 캐릭터 + 오브제 배치
                yield return SpawnSessionAssets(session, point, i);

                // PlazaSessionView 생성 (좋아요 UI + 말풍선)
                if (sessionViewPrefab != null)
                {
                    var view = Instantiate(sessionViewPrefab,
                        new Vector3(point.position.x, viewHeightOffset, point.position.z),
                        point.rotation * Quaternion.Euler(0f, 180f, 0f));
                    if (!view.gameObject.activeSelf) view.gameObject.SetActive(true);   // 비활성 프리팹 방어(코루틴/초기화 보장)
                    view.Initialize(session, session.session_id == _topSessionId);
                    // 이전 창작물은 캐릭터가 광장을 배회하므로, 투표 뷰(스폰 포인트 고정)에
                    // 캐릭터를 연결하지 않는다. 시그니처 재생은 PlazaWanderingCharacter 가 담당.
                    // (view.PlaySignatureMotion 은 캐릭터 미연결이라 자연히 no-op → 투표만 동작)
                    // 말풍선(BubblePanel)은 오브제 위로 분리 배치(카메라 빌보드)
                    view.PlaceBubbleAtObject(
                        _lastSpawnedObjectGO != null ? _lastSpawnedObjectGO.transform : null,
                        bubbleObjectHeightOffset, bubbleObjectScale, bubbleTextScale, bubbleObjectRightOffset);
                    // 좋아요 패널(LikePanel)은 오브제 '앞'(카메라 쪽)에 분리 배치(카메라 빌보드)
                    // forwardSizeFactor 로 오브제 크기에 따라 앞면 거리 자동 조정
                    view.PlaceLikePanelAtObject(
                        _lastSpawnedObjectGO != null ? _lastSpawnedObjectGO.transform : null,
                        likeObjectHeightOffset, likeObjectScale, likeObjectRightOffset,
                        likeObjectForwardOffset, likeObjectForwardSizeFactor);
                    _views.Add(view);
                }
                else
                {
                    Debug.LogWarning("[PlazaManager] sessionViewPrefab 미연결 — Inspector에서 연결하세요.");
                }
            }
        }

        private IEnumerator SpawnSessionAssets(PlazaSessionData session, Transform point, int sessionIndex)
        {
            // ── 캐릭터 ─────────────────────────────────────────────────────────
            var charData = session.assets?.character;
            if (charData != null)
            {
                GameObject charGo = null;

                if (charData.character_number > 0)
                {
                    charGo = SpawnPrebuiltCharacter(
                        session.session_id,
                        charData.character_number,
                        point);
                }
                else if (!string.IsNullOrEmpty(charData.model_url))
                {
                    // Server Mode: TriLib FBX 비동기 로드
                    bool done = false;
                    var wrapper = new GameObject($"PlazaChar_{session.session_id}");
                    wrapper.transform.SetPositionAndRotation(
                        point.position + point.forward * characterForwardOffset, point.rotation);
                    wrapper.transform.localScale = Vector3.one * characterSpawnScale;

                    TripoFbxLoader.Load(
                        charData.model_url, wrapper,
                        onLoaded: () => { charGo = wrapper; done = true; },
                        onError: msg =>
                        {
                            Debug.LogWarning($"[PlazaManager] 캐릭터 FBX 로드 실패: {msg} — Mock fallback");
                            Destroy(wrapper);
                            charGo = SpawnMockCharacter(session.session_id, point, sessionIndex);
                            done = true;
                        },
                        GetReferenceAvatar(sessionIndex));

                    yield return new WaitUntil(() => done);
                }
                else
                {
                    charGo = SpawnMockCharacter(session.session_id, point, sessionIndex);
                }

                if (charGo != null)
                {
                    SetupCharacterForPlaza(charGo, session.signature_motion, session.character_npc_name);
                    // 이전 창작물 캐릭터는 광장을 배회하며 접근 시 멈춰 시그니처 동작.
                    // (시그니처 트리거를 캐릭터 자신이 담당하므로 PlazaSessionView 와는 분리 — 투표는 그대로)
                    SetupWanderingCharacter(charGo);
                }
            }

            // ── 오브제 ──────────────────────────────────────────────────────────
            var objData = session.assets?.@object;
            if (objData != null)
            {
                _lastSpawnedObjectGO = null;   // 이번 세션 오브제 추적 초기화(말풍선 배치용)
                if (!string.IsNullOrEmpty(objData.model_url))
                {
                    // Server Mode: glTFast GLB 비동기 로드
                    var task = SpawnObjectFromServerAsync(session.session_id, objData, point, sessionIndex);
                    yield return new WaitUntil(() => task.IsCompleted);
                    if (task.IsFaulted)
                        Debug.LogError($"[PlazaManager] 광장 오브제 GLB 로드 실패: {task.Exception?.InnerException?.Message}");
                }
                else
                {
                    var objPrefab = PickMockPrefab(mockObjectPrefabs, sessionIndex);
                    if (objPrefab == null)
                        Debug.LogWarning("[PlazaManager] mockObjectPrefabs 미연결 — Inspector에서 연결하세요.");
                    else
                    {
                        var spawnPos = point.position - point.forward * objectBackOffset;
                        var objGo = Instantiate(objPrefab, spawnPos, point.rotation);
                        objGo.name = $"PlazaObj_{session.session_id}";
                        objGo.transform.localScale = Vector3.one * objectSpawnScale;
                        SetupObjectPhysics(objGo, point.position.y, objData.object_name);
                        _lastSpawnedObjectGO = objGo;
                    }
                }
            }
        }

        // 광장 캐릭터에 CharacterAnimationController + PlacedCharacterController 자동 주입.
        // 반환: 구성된 PlacedCharacterController (실패 시 null).
        private PlacedCharacterController SetupCharacterForPlaza(GameObject go, string signatureMotionName, string npcName)
        {
            // Animator Controller 설정 (없을 때만)
            if (characterAnimatorController != null)
            {
                var animator = go.GetComponentInChildren<Animator>();
                if (animator != null && animator.runtimeAnimatorController == null)
                    animator.runtimeAnimatorController = characterAnimatorController;
            }

            // CharacterAnimationController 먼저 — PlacedCharacterController가 Awake에서 참조
            var anim = go.GetComponent<CharacterAnimationController>();
            if (anim == null) anim = go.AddComponent<CharacterAnimationController>();

            var placed = go.GetComponent<PlacedCharacterController>();
            if (placed == null) placed = go.AddComponent<PlacedCharacterController>();

            // 저장된 시그니처는 "특정 클립 이름"(예: "Dance2"). 그 클립을 그대로 복원.
            // (옛 데이터의 MotionType 이름 "Dance"도 GetClipByName 이 호환 처리)
            var signatureClip = motionLibrary != null
                ? motionLibrary.GetClipByName(signatureMotionName)
                : null;

            placed.SetupForPlaza(motionLibrary, signatureClip);

            RuntimeOptimizer.Optimize(go);

            // 캐릭터 머리 위 이름표(있을 때만, 캐릭터용 축소 크기) — Mock·Server 공통
            NameTagSpawner.Instance?.AttachCharacter(go, npcName);
            return placed;
        }

        // 이전 창작물 캐릭터에 배회 + 근접 시그니처 인터랙션 컴포넌트를 부착한다.
        // (Mock·Server 공통 — SetupCharacterForPlaza 직후 호출)
        private void SetupWanderingCharacter(GameObject go)
        {
            var wander = go.GetComponent<PlazaWanderingCharacter>();
            if (wander == null) wander = go.AddComponent<PlazaWanderingCharacter>();

            // 기본: 각 캐릭터가 '자기 스폰 자리'를 중심으로 배회 → 스폰 지점이 흩어져 있어 자연 분산.
            // wanderCenter(Transform)를 지정하면 모든 캐릭터가 그 한 점을 공유(전역 공용 중심).
            bool useSpawnAsCenter = wanderCenter == null;
            Vector3 center = useSpawnAsCenter ? go.transform.position : wanderCenter.position;

            wander.Setup(center, wanderRadius, wanderMoveSpeed, approachRadius,
                         wanderWaitMin, wanderWaitMax, useSpawnAsCenter);
        }

        // 배열에서 index % length 순환 선택. 배열이 없거나 비어 있으면 null 반환.
        private static GameObject PickMockPrefab(GameObject[] prefabs, int index)
        {
            if (prefabs == null || prefabs.Length == 0) return null;
            return prefabs[index % prefabs.Length];
        }

        // Mock 캐릭터 프리팹을 스폰하고 GameObject를 반환한다. 프리팹 미연결 시 null.
        private GameObject SpawnMockCharacter(string sessionId, Transform point, int sessionIndex)
        {
            var prefab = PickMockPrefab(mockCharacterPrefabs, sessionIndex);
            if (prefab == null)
            {
                Debug.LogWarning("[PlazaManager] mockCharacterPrefabs 미연결 — Inspector에서 연결하세요.");
                return null;
            }
            var go = Instantiate(prefab,
                point.position + point.forward * characterForwardOffset, point.rotation);
            go.name = $"PlazaChar_{sessionId}";
            go.transform.localScale = Vector3.one * characterSpawnScale;
            return go;
        }

        private GameObject SpawnPrebuiltCharacter(string sessionId, int characterNumber, Transform point)
        {
            var resourcePath = $"{prebuiltCharacterResourcePrefix}{characterNumber}";
            var prefab = Resources.Load<GameObject>(resourcePath);
            if (prefab == null)
            {
                Debug.LogWarning($"[PlazaManager] 사전 제작 캐릭터 없음: Resources/{resourcePath}");
                return null;
            }

            var go = Instantiate(prefab,
                point.position + point.forward * characterForwardOffset, point.rotation);
            go.name = $"PlazaChar_{sessionId}_{characterNumber}";
            go.transform.localScale = Vector3.one * characterSpawnScale;
            return go;
        }

        private Avatar GetReferenceAvatar(int sessionIndex)
        {
            var prefab = PickMockPrefab(mockCharacterPrefabs, sessionIndex);
            if (prefab == null) return null;

            var animator = prefab.GetComponentInChildren<Animator>(true);
            if (animator == null) return null;

            var avatar = animator.avatar;
            return avatar != null && avatar.isValid && avatar.isHuman ? avatar : null;
        }

        // 오브제를 낙하 없이 groundY에 즉시 안착(Collider 재계산) + 전시용 렌더러 최적화
        // + 배회 캐릭터가 통과하지 않도록 NavMesh 장애물로 표시(오브제는 움직이지 않음)
        // + 오브제 머리 위 이름표(있을 때만). Mock/Server/fallback 3경로 공통 합류점.
        private static void SetupObjectPhysics(GameObject go, float groundY, string objectName)
        {
            RuntimePhysicsUtil.PlaceOnGround(go, groundY);
            RuntimeOptimizer.Optimize(go);
            AddNavMeshObstacle(go);
            NameTagSpawner.Instance?.Attach(go, objectName);
        }

        // 배회 캐릭터의 NavMeshAgent가 오브제를 피해 돌아가도록 장애물로 표시한다.
        // PlaceOnGround가 만든 루트 BoxCollider 크기를 그대로 사용(메시와 일치). 오브제는 정지 상태.
        private static void AddNavMeshObstacle(GameObject go)
        {
            if (go.GetComponent<NavMeshObstacle>() != null) return;

            var obs = go.AddComponent<NavMeshObstacle>();
            obs.carving             = true;    // NavMesh를 깎아 에이전트가 경로를 우회
            // 즉시 carve. true(정지 0.5초 대기 후 carve)이면 에이전트가 첫 경로를 잡은 뒤
            // 뒤늦게 carve 가 들어와 경로가 무효화되며 몇 초간 제자리에 멈추는 문제가 있었음.
            // 오브제는 스폰 즉시 고정(낙하 없음)이라 바로 carve 해도 추가 재계산이 없다.
            obs.carveOnlyStationary = false;

            var box = go.GetComponent<BoxCollider>();
            if (box != null)
            {
                obs.shape  = NavMeshObstacleShape.Box;
                obs.center = box.center;
                obs.size   = box.size;
            }
        }

        private async System.Threading.Tasks.Task SpawnObjectFromServerAsync(
            string sessionId, ObjectAssetData data, Transform point, int sessionIndex = 0)
        {
            var gltf = new GLTFast.GltfImport();
            bool ok = await gltf.Load(data.model_url);
            if (!ok)
            {
                Debug.LogWarning($"[PlazaManager] GLB 로드 실패: {data.model_url} — Mock fallback");
                var fallbackPrefab = PickMockPrefab(mockObjectPrefabs, sessionIndex);
                if (fallbackPrefab != null)
                {
                    var fallbackPos = point.position - point.forward * objectBackOffset;
                    var fallback = Instantiate(fallbackPrefab, fallbackPos, point.rotation);
                    fallback.name = $"PlazaObj_{sessionId}";
                    fallback.transform.localScale = Vector3.one * objectSpawnScale;
                    SetupObjectPhysics(fallback, point.position.y, data.object_name);
                    _lastSpawnedObjectGO = fallback;
                }
                return;
            }
            var spawnPos = point.position - point.forward * objectBackOffset;
            var root = new GameObject($"PlazaObj_{sessionId}");
            root.transform.SetPositionAndRotation(spawnPos, point.rotation);
            root.transform.localScale = Vector3.one * objectSpawnScale;
            await gltf.InstantiateMainSceneAsync(root.transform);
            // GLTFast 인스턴스화 직후 renderer.bounds가 미초기화 상태일 수 있으므로
            // 1프레임 대기 후 안착·BoxCollider 계산 (크기 0으로 잘못 안착하는 버그 방지)
            await Awaitable.NextFrameAsync();
            SetupObjectPhysics(root, point.position.y, data.object_name);
            _lastSpawnedObjectGO = root;
        }

        // ════════════════════════════════════════════════════════════
        // 좋아요 실시간 갱신
        // ════════════════════════════════════════════════════════════

        private void HandleLikesUpdated(WsLikesEvent ev)
        {
            // WebSocket push: 해당 세션만 즉시 업데이트
            string newTop = ev.top_session_id;
            foreach (var view in _views)
            {
                if (view == null) continue;
                if (view.SessionId == ev.session_id)
                    view.UpdateLikes(ev.likes);
                view.SetTopLiked(view.SessionId == newTop);
            }
            _topSessionId = newTop;
        }

        // ════════════════════════════════════════════════════════════
        // 현재 관람객 창작물 — 투표 뷰 부착
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 가이드 모드에서 이미 스폰된 오브제 위에 PlazaSessionView를 Instantiate해 투표 UI를 붙인다.
        /// sessionViewPrefab 또는 _currentObjectGO 가 없으면 조용히 스킵.
        /// </summary>
        private void AttachViewToCurrentSession()
        {
            if (_currentSession == null || sessionViewPrefab == null || _currentObjectGO == null)
                return;

            var sessionData = new PlazaSessionData
            {
                session_id         = _currentSession.session_id,
                character_npc_name = _currentSession.character_npc_name,
                bubble_text        = _currentSession.bubble_text,
                signature_motion   = _currentSession.signature_motion ?? "",
                character_number   = _currentSession.character_number,
                likes              = _currentSession.likes,
                is_top_liked       = _currentSession.session_id == _topSessionId,
                assets             = _currentSession.assets
            };

            _likeCounts[_currentSession.session_id] = _currentSession.likes;

            // 내 창작물의 동작 프롬프트 인터랙션(OwnCreationInteractor 트리거 + 패널)은 '캐릭터'
            // 근처에서 뜨도록, 뷰를 캐릭터 위치에 생성한다. 캐릭터를 오브제와 떨어뜨려 배치해도
            // 캐릭터에 다가가면 UI가 뜬다. (말풍선은 PlaceBubbleAtObject 로 오브제 위에 별도 배치.)
            // 캐릭터가 없으면(드문 경우) 오브제 위치로 폴백.
            var anchorGo  = _currentCharacterGO != null ? _currentCharacterGO : _currentObjectGO;
            var anchorPos = anchorGo.transform.position;
            var pos = new Vector3(anchorPos.x, viewHeightOffset, anchorPos.z);
            var rot = anchorGo.transform.rotation * Quaternion.Euler(0f, 180f, 0f);
            var view = Instantiate(sessionViewPrefab, pos, rot);
            if (!view.gameObject.activeSelf) view.gameObject.SetActive(true);   // 비활성 프리팹 방어(코루틴/초기화 보장)
            view.Initialize(sessionData, sessionData.is_top_liked);
            _views.Add(view);

            // ── 내 창작물 전용 설정 ────────────────────────────────────────

            // 자신의 창작물에는 좋아요 버튼 숨김 (자기 자신에게 투표 불가)
            var likeSystem = view.GetComponent<LikeSystem>();
            if (likeSystem != null) likeSystem.likePanel = null;

            // 배치 캐릭터 연결: 시그니처 동작 재생 + OwnCreationInteractor 의 모션 대상
            var placedChar = _currentCharacterGO?.GetComponent<PlacedCharacterController>();
            view.SetCharacter(placedChar);

            // 동작 프롬프트 · 시그니처 설정 인터랙터 추가
            SetupOwnCreationInteractor(view.gameObject, placedChar);

            // 말풍선(BubblePanel)은 내 오브제 위로 분리 배치(카메라 빌보드)
            view.PlaceBubbleAtObject(_currentObjectGO.transform, bubbleObjectHeightOffset, bubbleObjectScale, bubbleTextScale, bubbleObjectRightOffset);

            _currentViewAttached = true;
        }

        private void SetupOwnCreationInteractor(GameObject viewGo, PlacedCharacterController placedChar)
        {
            // Inspector 미연결 시 씬에서 자동 검색 (GameFlowManager에 연결된 것과 동일 컴포넌트 사용)
            var promptUI  = motionPromptUI   != null ? motionPromptUI   : FindAnyObjectByType<MotionPromptUI>();
            var confirmUI = signatureConfirmUI != null ? signatureConfirmUI : FindAnyObjectByType<SignatureMotionConfirmUI>();

            if (promptUI == null)
                Debug.LogWarning("[PlazaManager] MotionPromptUI를 찾을 수 없습니다. " +
                                 "씬에 MotionPromptUI GameObject가 있는지 확인하세요.");

            var interactor = viewGo.AddComponent<OwnCreationInteractor>();

            GameObject panel = null;
            if (ownCreationInteractPanelPrefab != null)
            {
                // 패널은 자체 World Space Canvas + GraphicRaycaster(버튼 클릭)라 reparent-stretch 하면
                // 안 됨. 좋아요 패널과 동일하게 '캐릭터 앞·눈높이'로 따라가도록 독립 생성 후 NameTag 부착.
                // (뷰의 빌보드와 충돌 방지 위해 뷰의 자식으로 두지 않음. NameTag 가 위치/빌보드 담당.)
                panel = Instantiate(ownCreationInteractPanelPrefab);
                panel.transform.localScale *= interactPanelScale;   // 1=프리팹 원본(기본), 그 외 배율 적용

                var cam = Camera.main;
                foreach (var c in panel.GetComponentsInChildren<Canvas>(true))
                    if (c.renderMode == RenderMode.WorldSpace) c.worldCamera = cam;

                var target = placedChar != null ? placedChar.transform : viewGo.transform;
                // 동작 입력 패널 전용 오프셋으로 캐릭터 앞·눈높이에 배치(좋아요 패널과 독립 조정).
                // 높이는 눈높이 기준이며, interactPanelHeightOffset(음수)으로 눈높이보다 낮춘다.
                panel.AddComponent<NameTag>().Bind(target,
                    interactPanelHeightOffset, interactPanelRightOffset,
                    interactPanelForwardOffset, interactPanelForwardSizeFactor, eyeLevel: true);
            }

            interactor.Setup(placedChar, panel, promptUI, confirmUI);
        }

        // ════════════════════════════════════════════════════════════
        // Server 모드 — 좋아요 즉시 반영 (LikeResponse 기반)
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// Server 모드에서 POST /like 응답 수신 시 LikeSystem이 호출.
        /// 해당 세션 좋아요 수와 모든 세션의 별 표시를 즉시 갱신한다.
        /// </summary>
        public void HandleServerLike(LikeResponse result)
        {
            if (result == null) return;
            string newTop = result.top_session_id;
            foreach (var view in _views)
            {
                if (view == null) continue;
                if (view.SessionId == result.session_id)
                    view.UpdateLikes(result.likes);
                view.SetTopLiked(view.SessionId == newTop);
            }
            _topSessionId = newTop;
        }

        // ════════════════════════════════════════════════════════════
        // Mock 모드 — 좋아요 로컬 처리
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// Mock 모드에서 LikeSystem이 좋아요를 눌렀을 때 호출.
        /// 딕셔너리에서 해당 세션 좋아요 수를 갱신하고 1위 세션을 재계산한다.
        /// </summary>
        public void HandleMockLike(string sessionId, int newLikes)
        {
            _likeCounts[sessionId] = newLikes;

            // 가장 좋아요가 많은 세션 ID 계산
            string newTop = _likeCounts.OrderByDescending(kv => kv.Value).First().Key;

            foreach (var view in _views)
            {
                if (view == null) continue;
                view.SetTopLiked(view.SessionId == newTop);
            }
            _topSessionId = newTop;

            SaveMockLikesToFile(sessionId, newLikes, newTop);
        }

        private void SaveMockLikesToFile(string sessionId, int newLikes, string topSessionId)
        {
            try
            {
                var plaza = MockPlazaStore.Load();
                if (plaza?.sessions == null)
                {
                    Debug.LogWarning("[SaveMockLikes] 역직렬화 실패 — sessions null");
                    return;
                }

                var target = plaza.sessions.Find(s => s.session_id == sessionId);
                if (target == null) return;  // 현재 세션은 아직 파일에 없음 — 종료 시 저장됨

                target.likes = newLikes;
                plaza.top_session_id = topSessionId;

                foreach (var s in plaza.sessions)
                    s.is_top_liked = s.session_id == topSessionId;

                MockPlazaStore.Save(plaza);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[SaveMockLikes] 파일 저장 실패: {e.Message}");
            }
        }

    }
}
