using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using LegoTwin.Character;
using LegoTwin.Data;
using LegoTwin.Managers;
using LegoTwin.Network;
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

        [Header("스폰 설정")]
        public float characterSpawnScale    = 6f;
        public float objectSpawnScale       = 12f;
        public float characterForwardOffset = 3f;   // 캐릭터: spawnPoint 앞쪽
        public float objectBackOffset       = 3f;   // 오브제:  spawnPoint 뒤쪽
        public float viewHeightOffset       = 5f;   // 투표 UI: spawnPoint에서 위로 띄울 높이

        [Header("오브제 낙하 설정")]
        [Tooltip("오브제가 스폰되는 높이 (이 높이에서 중력으로 낙하)")]
        public float objectDropHeight = 15f;

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

        private readonly List<PlazaSessionView> _views = new();
        private readonly Dictionary<string, int> _likeCounts = new();
        private string _topSessionId;
        private PlacedCharacterController _lastSpawnedCharController;

        // 현재 관람객 창작물 (가이드 모드에서 스폰된 오브젝트)
        private SessionData _currentSession;
        private GameObject  _currentCharacterGO;
        private GameObject  _currentObjectGO;
        private bool        _currentViewAttached;   // 현재 창작물 뷰 부착 완료 여부 (중복 방지)

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
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
                    view.Initialize(session, session.session_id == _topSessionId);
                    view.SetCharacter(_lastSpawnedCharController);
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
                // FBX 런타임 로드는 TriLib 미구현 → model_url 유무와 관계없이 Mock fallback
                if (!string.IsNullOrEmpty(charData.model_url))
                    Debug.LogWarning($"[PlazaManager] 캐릭터 서버 로드 미구현 ({charData.model_url}) — Mock fallback");

                var charPrefab = PickMockPrefab(mockCharacterPrefabs, sessionIndex);
                if (charPrefab == null)
                    Debug.LogWarning("[PlazaManager] mockCharacterPrefabs 미연결 — Inspector에서 연결하세요.");
                else
                {
                    var charGo = Instantiate(charPrefab,
                        point.position + point.forward * characterForwardOffset, point.rotation);
                    charGo.name = $"PlazaChar_{session.session_id}";
                    charGo.transform.localScale = Vector3.one * characterSpawnScale;
                    SetupCharacterForPlaza(charGo, session.signature_motion);
                }
            }

            // ── 오브제 ──────────────────────────────────────────────────────────
            var objData = session.assets?.@object;
            if (objData != null)
            {
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
                        var spawnPos = point.position
                                       - point.forward * objectBackOffset
                                       + Vector3.up * objectDropHeight;
                        var objGo = Instantiate(objPrefab, spawnPos, point.rotation);
                        objGo.name = $"PlazaObj_{session.session_id}";
                        objGo.transform.localScale = Vector3.one * objectSpawnScale;
                        SetupObjectPhysics(objGo);
                    }
                }
            }
        }

        // 광장 캐릭터에 CharacterAnimationController + PlacedCharacterController 자동 주입
        private void SetupCharacterForPlaza(GameObject go, string signatureMotionName)
        {
            _lastSpawnedCharController = null;

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

            var motionType = MotionType.Idle;
            if (!string.IsNullOrEmpty(signatureMotionName) &&
                System.Enum.TryParse(signatureMotionName, out MotionType parsed))
                motionType = parsed;

            placed.SetupForPlaza(motionLibrary, motionType);
            _lastSpawnedCharController = placed;
        }

        // 배열에서 index % length 순환 선택. 배열이 없거나 비어 있으면 null 반환.
        private static GameObject PickMockPrefab(GameObject[] prefabs, int index)
        {
            if (prefabs == null || prefabs.Length == 0) return null;
            return prefabs[index % prefabs.Length];
        }

        // 오브제에 Rigidbody(중력) + Collider 자동 부착
        private static void SetupObjectPhysics(GameObject go)
        {
            var rb = go.GetComponent<Rigidbody>();
            if (rb == null) rb = go.AddComponent<Rigidbody>();
            rb.useGravity  = true;
            rb.constraints = RigidbodyConstraints.FreezeRotation;

            // 기존 Collider 없을 때만 MeshCollider 자동 추가
            if (go.GetComponentInChildren<Collider>() == null)
            {
                foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
                {
                    if (mf.sharedMesh == null) continue;
                    var col = mf.gameObject.GetComponent<MeshCollider>();
                    if (col == null) col = mf.gameObject.AddComponent<MeshCollider>();
                    col.sharedMesh = mf.sharedMesh;
                    col.convex     = true;   // Dynamic Rigidbody 필수
                }
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
                    var fallbackPos = point.position
                                      - point.forward * objectBackOffset
                                      + Vector3.up * objectDropHeight;
                    var fallback = Instantiate(fallbackPrefab, fallbackPos, point.rotation);
                    fallback.name = $"PlazaObj_{sessionId}";
                    fallback.transform.localScale = Vector3.one * objectSpawnScale;
                    SetupObjectPhysics(fallback);
                }
                return;
            }
            var dropPos = point.position - point.forward * objectBackOffset + Vector3.up * objectDropHeight;
            var root = new GameObject($"PlazaObj_{sessionId}");
            root.transform.SetPositionAndRotation(dropPos, point.rotation);
            root.transform.localScale = Vector3.one * objectSpawnScale;
            await gltf.InstantiateMainSceneAsync(root.transform);
            SetupObjectPhysics(root);
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
                likes              = _currentSession.likes,
                is_top_liked       = _currentSession.session_id == _topSessionId,
                assets             = _currentSession.assets
            };

            _likeCounts[_currentSession.session_id] = _currentSession.likes;

            // 이전 관람객 뷰와 동일한 절대 Y 기준 사용 (object Y 무시)
            var objPos = _currentObjectGO.transform.position;
            var pos = new Vector3(objPos.x, viewHeightOffset, objPos.z);
            var rot = _currentObjectGO.transform.rotation * Quaternion.Euler(0f, 180f, 0f);
            var view = Instantiate(sessionViewPrefab, pos, rot);
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
                panel = Instantiate(ownCreationInteractPanelPrefab, viewGo.transform);

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
