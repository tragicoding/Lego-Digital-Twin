using System;
using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;
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
            transform.SetParent(null);   // 자식 오브젝트면 루트로 분리
            DontDestroyOnLoad(gameObject);

            // ApiClient 자동 연결
            if (_apiClient == null)
                _apiClient = FindAnyObjectByType<ApiClient>();
        }

        private void Start()
        {
            // Server Mode이고 Inspector에 sessionId가 비어 있으면 큐에서 자동 감지
            if (dataSourceMode == DataSourceMode.Server && string.IsNullOrEmpty(sessionId))
                StartCoroutine(AutoDetectAndLoad());
            else
                LoadSession(sessionId);
        }

        private IEnumerator AutoDetectAndLoad()
        {
            if (_apiClient == null)
            {
                Debug.LogError("[SessionManager] ApiClient 없음 — 씬에 ApiClient를 추가하세요.");
                yield break;
            }

            string sid = null;
            yield return _apiClient.FetchActiveSession(s => sid = s);

            if (!string.IsNullOrEmpty(sid))
            {
                sessionId = sid;
                LoadSession(sid);
            }
            else
            {
                Debug.LogWarning("[SessionManager] 대기 큐 비어있음 — 다음 세션 대기 중...");
                StartCoroutine(PollForNextSession());
            }
        }

        // 큐에 다음 세션이 생길 때까지 5초마다 폴링
        private IEnumerator PollForNextSession(float interval = 5f)
        {
            while (true)
            {
                yield return new WaitForSeconds(interval);
                string sid = null;
                yield return _apiClient.FetchActiveSession(s => sid = s);
                if (!string.IsNullOrEmpty(sid))
                {
                    sessionId = sid;
                    LoadSession(sid);
                    yield break;
                }
            }
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

                StartCoroutine(LoadServerSession(sid, onLoaded));
            }
        }

        private IEnumerator LoadServerSession(string sid, Action<SessionData> onLoaded)
        {
            yield return _apiClient.PollUntilReady(sid, null);
            yield return _apiClient.FetchUnitySession(sid, data => Apply(data, onLoaded));
        }

        private void Apply(SessionData data, Action<SessionData> onLoaded)
        {
            if (data == null)
            {
                Debug.LogError("[SessionManager] SessionData 로드 실패");
                return;
            }
            CurrentSession = data;
            onLoaded?.Invoke(data);
            OnSessionLoaded?.Invoke(data);
        }

        // ════════════════════════════════════════════════════════════
        // 시그니처 동작 저장 — 가이드 모드 · 자유 모드 공통 경로
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 현재 세션의 시그니처 동작을 설정한다.
        /// signatureClipName = 관람객이 고른 특정 클립 이름(예: "Dance2").
        /// Mock: 메모리에만 반영 (앱 종료 시 PlazaManager가 파일로 저장).
        /// Server: 메모리 반영 + PATCH /sessions/{id}/signature-motion API 호출.
        /// </summary>
        public void SetSignatureMotion(string signatureClipName)
        {
            if (CurrentSession == null)
            {
                Debug.LogWarning("[SessionManager] 시그니처 동작 저장 실패 — 현재 세션 없음");
                return;
            }

            CurrentSession.signature_motion = signatureClipName;

            if (dataSourceMode == DataSourceMode.Server && _apiClient != null)
                StartCoroutine(_apiClient.SaveSignatureMotion(CurrentSession.session_id, signatureClipName));
        }

        // ════════════════════════════════════════════════════════════
        // 종료 — 큐 전진 + 다음 세션 로드
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 종료 버튼 호출 진입점.
        /// Mock: plaza 저장 후 씬 리로드.
        /// Server: 큐 전진 → 다음 세션 있으면 씬 리로드, 없으면 onNoNext 호출(앱 종료 등).
        /// </summary>
        public void AdvanceAndLoadNext(Action onNoNext = null)
        {
            if (dataSourceMode == DataSourceMode.Mock)
            {
                LegoTwin.Plaza.PlazaManager.Instance?.SaveCurrentSessionToMockPlaza();
                ReloadScene();
                return;
            }
            StartCoroutine(AdvanceAndLoadNextCoroutine(onNoNext));
        }

        private IEnumerator AdvanceAndLoadNextCoroutine(Action onNoNext)
        {
            if (_apiClient == null) { onNoNext?.Invoke(); yield break; }

            string nextSid = null;
            yield return _apiClient.AdvanceQueue(s => nextSid = s);

            if (!string.IsNullOrEmpty(nextSid))
            {
                // 다음 사용자 세션으로 전환 — 씬 리로드로 오브젝트 초기화
                sessionId      = nextSid;
                CurrentSession = null;
                LoadSession(nextSid, _ => ReloadScene());
            }
            else
            {
                // 대기 큐 비어있음 → 호출부가 앱 종료 등 처리
                onNoNext?.Invoke();
            }
        }

        private static void ReloadScene() =>
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }
}
