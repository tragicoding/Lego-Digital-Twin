using System;
using System.Collections;
using UnityEngine;
using LegoTwin.Data;
using LegoTwin.Mock;
using LegoTwin.Network;

namespace LegoTwin.Managers
{
    /// <summary>
    /// 세션 데이터 로드 진입점.
    ///
    /// Inspector에서 dataSourceMode를 선택:
    ///   Mock   → 서버 없이 mock_session.json 사용 (기본값, 팀원 개발용)
    ///   Server → FastAPI 서버에서 실제 데이터 수신 (전시 통합용)
    ///
    /// 세션 로드 완료 시 OnSessionLoaded 이벤트 발생:
    ///   SessionManager.Instance.OnSessionLoaded += MyHandler;
    /// </summary>
    public class SessionManager : MonoBehaviour
    {
        public static SessionManager Instance { get; private set; }

        [Header("Mode")]
        [Tooltip("Mock: 서버 없이 JSON 사용 / Server: FastAPI 연동")]
        public DataSourceMode dataSourceMode = DataSourceMode.Mock;

        [Header("Server Mode")]
        [Tooltip("Server Mode 사용 시 session_id 입력")]
        public string sessionId;

        [Header("Components")]
        [Tooltip("Server Mode에서만 사용. 씬에 ApiClient가 있으면 자동 연결.")]
        [SerializeField] private ApiClient _apiClient;

        public SessionData CurrentSession { get; private set; }

        /// <summary>세션 로드 완료 시 호출된다.</summary>
        public event Action<SessionData> OnSessionLoaded;

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            // ApiClient 자동 연결
            if (_apiClient == null)
                _apiClient = FindObjectOfType<ApiClient>();
        }

        private void Start()
        {
            LoadSession(sessionId);
        }

        /// <summary>현재 모드에 따라 세션 데이터를 로드한다.</summary>
        public void LoadSession(string sid = null, Action<SessionData> onLoaded = null)
        {
            if (dataSourceMode == DataSourceMode.Mock)
            {
                var data = MockSessionLoader.Load();
                Apply(data, onLoaded);
            }
            else
            {
                if (string.IsNullOrEmpty(sid)) sid = sessionId;

                if (_apiClient == null)
                {
                    Debug.LogError("[SessionManager] Server Mode이지만 ApiClient를 찾을 수 없습니다. 씬에 ApiClient를 추가하세요.");
                    return;
                }

                StartCoroutine(_apiClient.FetchUnitySession(sid, data => Apply(data, onLoaded)));
            }
        }

        private void Apply(SessionData data, Action<SessionData> onLoaded)
        {
            if (data == null)
            {
                Debug.LogError("[SessionManager] SessionData 로드 실패");
                return;
            }
            CurrentSession = data;
            Debug.Log($"[SessionManager] 로드 완료 — 캐릭터: {data.character_npc_name}, 오브제: {data.object_name}");
            onLoaded?.Invoke(data);
            OnSessionLoaded?.Invoke(data);
        }
    }
}
