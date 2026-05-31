using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using LegoTwin.Data;
using LegoTwin.Managers;
using LegoTwin.Network;

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

        [Header("배치 위치 (캐릭터+오브제 쌍, 순서대로)")]
        public Transform[] spawnPoints;

        [Header("Mock Mode Prefab")]
        public GameObject mockCharacterPrefab;
        public GameObject mockObjectPrefab;

        [Header("스폰 설정")]
        public float spawnScale = 12f;

        [Header("PlazaSessionView 프리팹 (UI + 좋아요 포함)")]
        public PlazaSessionView sessionViewPrefab;

        private readonly List<PlazaSessionView> _views = new();
        private readonly Dictionary<string, int> _likeCounts = new();
        private string _topSessionId;

        // 현재 관람객 창작물 (가이드 모드에서 스폰된 오브젝트)
        private SessionData _currentSession;
        private GameObject  _currentCharacterGO;
        private GameObject  _currentObjectGO;

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
            _currentSession     = session;
            _currentCharacterGO = charGo;
            _currentObjectGO    = objGo;
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

        // ════════════════════════════════════════════════════════════
        // 로드 + 배치
        // ════════════════════════════════════════════════════════════

        private IEnumerator LoadAndSpawnPlaza()
        {
            PlazaResponse plaza = null;

            if (SessionManager.Instance?.dataSourceMode == Managers.DataSourceMode.Mock)
            {
                plaza = LoadMockPlaza();
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
            Debug.Log($"[PlazaManager] 광장 데이터 로드 완료 — 세션 수: {plaza.sessions.Count}, 1위: {plaza.top_session_id}, spawnPoints: {spawnPoints.Length}");

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

                Debug.Log($"[PlazaManager] 세션 배치 [{i}]: {session.session_id} / {session.character_npc_name} → {point.name}");

                _likeCounts[session.session_id] = session.likes;

                // 캐릭터 + 오브제 배치
                yield return SpawnSessionAssets(session, point);

                // PlazaSessionView 생성 (좋아요 UI + 말풍선)
                if (sessionViewPrefab != null)
                {
                    var view = Instantiate(sessionViewPrefab, point.position, point.rotation);
                    view.Initialize(session, session.session_id == _topSessionId);
                    _views.Add(view);
                }
                else
                {
                    Debug.LogWarning("[PlazaManager] sessionViewPrefab 미연결 — Inspector에서 연결하세요.");
                }
            }
        }

        private IEnumerator SpawnSessionAssets(PlazaSessionData session, Transform point)
        {
            // ── 캐릭터 ─────────────────────────────────────────────────────────
            var charData = session.assets?.character;
            if (charData != null)
            {
                // FBX 런타임 로드는 TriLib 미구현 → model_url 유무와 관계없이 Mock fallback
                if (!string.IsNullOrEmpty(charData.model_url))
                    Debug.LogWarning($"[PlazaManager] 캐릭터 서버 로드 미구현 ({charData.model_url}) — Mock fallback");

                if (mockCharacterPrefab == null)
                    Debug.LogWarning("[PlazaManager] mockCharacterPrefab 미연결 — Inspector에서 연결하세요.");
                else
                {
                    var charGo = Instantiate(mockCharacterPrefab,
                        point.position + Vector3.left * 3f, point.rotation);
                    charGo.name = $"PlazaChar_{session.session_id}";
                    charGo.transform.localScale = Vector3.one * spawnScale;
                    Debug.Log($"[PlazaManager] 캐릭터 스폰: {charGo.name}");
                }
            }

            // ── 오브제 ──────────────────────────────────────────────────────────
            var objData = session.assets?.@object;
            if (objData != null)
            {
                if (!string.IsNullOrEmpty(objData.model_url))
                {
                    // Server Mode: glTFast GLB 비동기 로드
                    var task = SpawnObjectFromServerAsync(session.session_id, objData, point);
                    yield return new WaitUntil(() => task.IsCompleted);
                    if (task.IsFaulted)
                        Debug.LogError($"[PlazaManager] 광장 오브제 GLB 로드 실패: {task.Exception?.InnerException?.Message}");
                }
                else
                {
                    if (mockObjectPrefab == null)
                        Debug.LogWarning("[PlazaManager] mockObjectPrefab 미연결 — Inspector에서 연결하세요.");
                    else
                    {
                        var objGo = Instantiate(mockObjectPrefab,
                            point.position + Vector3.right * 3f, point.rotation);
                        objGo.name = $"PlazaObj_{session.session_id}";
                        objGo.transform.localScale = Vector3.one * spawnScale;
                        Debug.Log($"[PlazaManager] 오브제 스폰: {objGo.name}");
                    }
                }
            }
        }

        private async System.Threading.Tasks.Task SpawnObjectFromServerAsync(
            string sessionId, ObjectAssetData data, Transform point)
        {
            Debug.Log($"[PlazaManager] 광장 오브제 GLB 로드: {data.model_url}");
            var gltf = new GLTFast.GltfImport();
            bool ok = await gltf.Load(data.model_url);
            if (!ok)
            {
                Debug.LogWarning($"[PlazaManager] GLB 로드 실패: {data.model_url} — Mock fallback");
                if (mockObjectPrefab != null)
                {
                    var fallback = Instantiate(mockObjectPrefab,
                        point.position + Vector3.right * 3f, point.rotation);
                    fallback.name = $"PlazaObj_{sessionId}";
                    fallback.transform.localScale = Vector3.one * spawnScale;
                }
                return;
            }
            var root = new GameObject($"PlazaObj_{sessionId}");
            root.transform.SetPositionAndRotation(point.position + Vector3.right * 3f, point.rotation);
            root.transform.localScale = Vector3.one * spawnScale;
            await gltf.InstantiateMainSceneAsync(root.transform);
            Debug.Log($"[PlazaManager] 광장 오브제 서버 스폰 완료: {root.name}");
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
            {
                Debug.Log("[PlazaManager] 현재 세션 뷰 부착 생략 " +
                          "(currentSession/sessionViewPrefab/objectGO 중 하나 없음)");
                return;
            }

            var sessionData = new PlazaSessionData
            {
                session_id         = _currentSession.session_id,
                character_npc_name = _currentSession.character_npc_name,
                bubble_text        = _currentSession.bubble_text,
                likes              = _currentSession.likes,
                is_top_liked       = _currentSession.session_id == _topSessionId,
                assets             = _currentSession.assets
            };

            // 오브제 위치 바로 위에 뷰 배치 (Y 오프셋은 Inspector에서 spawnPoint로 조정)
            _likeCounts[_currentSession.session_id] = _currentSession.likes;

            var pos = _currentObjectGO.transform.position;
            var rot = _currentObjectGO.transform.rotation;
            var view = Instantiate(sessionViewPrefab, pos, rot);
            view.Initialize(sessionData, sessionData.is_top_liked);
            _views.Add(view);

            Debug.Log($"[PlazaManager] 현재 세션 뷰 부착 완료: {_currentSession.session_id}");
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
            Debug.Log($"[PlazaManager] 서버 좋아요 반영: {result.session_id} → {result.likes}, 1위: {newTop}");
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
        }

        // ════════════════════════════════════════════════════════════
        // Mock 데이터
        // ════════════════════════════════════════════════════════════

        private static PlazaResponse LoadMockPlaza()
        {
            var asset = Resources.Load<TextAsset>("Mock/mock_plaza");
            if (asset == null)
            {
                Debug.LogError("[PlazaManager] Resources/Mock/mock_plaza.json 없음");
                return null;
            }
            return JsonUtility.FromJson<PlazaResponse>(asset.text);
        }
    }
}
