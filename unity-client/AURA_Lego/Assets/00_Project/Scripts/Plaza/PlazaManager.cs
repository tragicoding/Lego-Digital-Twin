using System.Collections;
using System.Collections.Generic;
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

        [Header("PlazaSessionView 프리팹 (UI + 좋아요 포함)")]
        public PlazaSessionView sessionViewPrefab;

        private readonly List<PlazaSessionView> _views = new();
        private string _topSessionId;

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
        }

        // ════════════════════════════════════════════════════════════
        // 진입점 — 가이드 모드 완료 후 호출
        // ════════════════════════════════════════════════════════════

        /// <summary>자유 모드 진입 시 호출. 광장 데이터 로드 + 배치 시작.</summary>
        public void EnterPlaza()
        {
            StartCoroutine(LoadAndSpawnPlaza());

            // 실시간 좋아요 구독
            if (WebSocketManager.Instance != null)
                WebSocketManager.Instance.OnLikesUpdated += HandleLikesUpdated;
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

            for (int i = 0; i < plaza.sessions.Count; i++)
            {
                if (i >= spawnPoints.Length)
                {
                    Debug.LogWarning("[PlazaManager] spawnPoints 부족 — Inspector에 더 추가하세요.");
                    break;
                }

                var session = plaza.sessions[i];
                var point   = spawnPoints[i];

                // 캐릭터 + 오브제 배치
                SpawnSessionAssets(session, point);

                // PlazaSessionView 생성 (좋아요 UI + 말풍선)
                if (sessionViewPrefab != null)
                {
                    var view = Instantiate(sessionViewPrefab, point.position, point.rotation);
                    view.Initialize(session, session.session_id == _topSessionId);
                    _views.Add(view);
                }
            }
        }

        private void SpawnSessionAssets(PlazaSessionData session, Transform point)
        {
            // 캐릭터
            var charData = session.assets?.character;
            if (charData != null)
            {
                var prefab = string.IsNullOrEmpty(charData.model_url) ? mockCharacterPrefab : mockCharacterPrefab;
                // TODO: Server Mode — model_url로 런타임 FBX 로드 (TriLib)
                if (prefab != null)
                {
                    var charGo = Instantiate(prefab,
                        point.position + Vector3.left * 0.5f, point.rotation);
                    charGo.name = $"PlazaChar_{session.session_id}";
                }
            }

            // 오브제
            var objData = session.assets?.@object;
            if (objData != null)
            {
                var prefab = string.IsNullOrEmpty(objData.model_url) ? mockObjectPrefab : mockObjectPrefab;
                // TODO: Server Mode — model_url로 glTFast GLB 로드
                if (prefab != null)
                {
                    var objGo = Instantiate(prefab,
                        point.position + Vector3.right * 0.5f, point.rotation);
                    objGo.name = $"PlazaObj_{session.session_id}";
                }
            }
        }

        // ════════════════════════════════════════════════════════════
        // 좋아요 실시간 갱신
        // ════════════════════════════════════════════════════════════

        private void HandleLikesUpdated(WsLikesEvent ev)
        {
            if (ev.@event == "poll_tick")
            {
                // 폴링 방식: 전체 다시 로드 (NativeWebSocket 미사용 시)
                StartCoroutine(RefreshPlaza());
                return;
            }

            // WebSocket 방식: 해당 세션만 업데이트
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

        private IEnumerator RefreshPlaza()
        {
            if (SessionManager.Instance?.dataSourceMode == Managers.DataSourceMode.Mock)
                yield break;

            PlazaResponse plaza = null;
            yield return ApiClient.Instance.FetchPlazaSessions(r => plaza = r);
            if (plaza?.sessions == null) yield break;

            _topSessionId = plaza.top_session_id;
            foreach (var session in plaza.sessions)
            {
                var view = _views.Find(v => v != null && v.SessionId == session.session_id);
                if (view == null) continue;
                view.UpdateLikes(session.likes);
                view.SetTopLiked(session.session_id == _topSessionId);
            }
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
