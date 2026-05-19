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
    /// Inspector에서 dataSourceMode를 선택한다:
    ///   - Mock   : 서버 없이 mock_session.json 사용 (기본값)
    ///   - Server : FastAPI 서버에서 실제 데이터 수신
    ///
    /// 사용 예:
    ///   SessionManager.Instance.LoadSession(sessionId, OnSessionReady);
    /// </summary>
    public class SessionManager : MonoBehaviour
    {
        public static SessionManager Instance { get; private set; }

        [Header("Mode")]
        [Tooltip("Mock: 서버 없이 로컬 JSON 사용 / Server: FastAPI 서버 연동")]
        public DataSourceMode dataSourceMode = DataSourceMode.Mock;

        [Header("Server Mode")]
        [Tooltip("Server Mode에서 사용할 session_id (Inspector 직접 입력 또는 외부에서 주입)")]
        public string sessionId;

        public SessionData CurrentSession { get; private set; }

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        private void Start()
        {
            LoadSession(sessionId);
        }

        /// <summary>현재 모드에 따라 세션 데이터를 로드한다.</summary>
        public void LoadSession(string sid = null, System.Action<SessionData> onLoaded = null)
        {
            if (dataSourceMode == DataSourceMode.Mock)
            {
                var data = MockSessionLoader.Load();
                CurrentSession = data;
                Debug.Log($"[SessionManager] Mock 데이터 로드 완료: {data?.character_npc_name}");
                onLoaded?.Invoke(data);
            }
            else
            {
                if (string.IsNullOrEmpty(sid)) sid = sessionId;
                StartCoroutine(ApiClient.Instance.FetchUnitySession(sid, data =>
                {
                    CurrentSession = data;
                    Debug.Log($"[SessionManager] 서버 데이터 로드 완료: {data?.character_npc_name}");
                    onLoaded?.Invoke(data);
                }));
            }
        }
    }
}
