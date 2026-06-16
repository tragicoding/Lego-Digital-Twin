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
            BeginSessionAcquisition();
        }

        /// <summary>
        /// 세션 획득을 시작한다. 앱 첫 시작과 (A안) 종료 후 씬 리로드 재진입에서 공통 사용한다.
        /// SessionManager는 DontDestroyOnLoad라 리로드해도 Start가 재실행되지 않으므로,
        /// 리로드 후에는 이 메서드를 명시적으로 다시 호출해 다음 세션을 받는다.
        ///   Server + sessionId 비어있음 → 대기 큐 자동 감지/폴링 (대기화면 유지)
        ///   그 외(지정 sessionId·Mock)   → 즉시 로드
        /// </summary>
        private void BeginSessionAcquisition()
        {
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

            // object_name / bubble_text 는 전시 흐름상 같은 의미로 사용한다.
            // 레거시/서버 응답 차이를 흡수해 Unity 내부에서는 두 필드가 항상 동기화되게 맞춘다.
            if (string.IsNullOrEmpty(data.bubble_text) && !string.IsNullOrEmpty(data.object_name))
                data.bubble_text = data.object_name;
            if (string.IsNullOrEmpty(data.object_name) && !string.IsNullOrEmpty(data.bubble_text))
                data.object_name = data.bubble_text;

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

        /// <summary>
        /// 현재 세션의 캐릭터/오브제 이름을 설정한다(대기화면 이름 입력).
        /// 이름 입력은 백엔드가 아니라 Unity(대기화면)에서 받는다 — 입력의 단일 출처는 Unity다.
        /// 빈 값은 무시(기존 이름 유지)하므로, 입력을 건너뛰어도 안전하다.
        /// 스폰 '이전'에 호출되므로 가이드 NPC 대사·배치 캐릭터/오브제 이름표에 처음부터 반영된다(사후 갱신 불필요).
        ///
        /// object_name 은 오브제 '이름표'용 필드(assets.object.object_name)만 갱신하고,
        /// 한마디(bubble_text)와 공유되는 최상위 object_name 은 건드리지 않는다(이름 ≠ 한마디).
        ///
        /// 영구 저장(광장에서 '이전 창작물 이름'을 표시하기 위함):
        ///   Mock  : 메모리 반영 → 세션 완료 시 PlazaManager가 mock_plaza.json에 저장(assets 포함).
        ///   Server: 메모리 반영 + PATCH /sessions/{id}/names → 백엔드가 보관 → plaza API가 다음 관람객에게 반환.
        /// caller(GameFlowManager)는 분기 없이 이 한 경로만 호출한다(SetBubbleText와 동일 패턴).
        /// </summary>
        public void SetNames(string characterName, string objectName)
        {
            if (CurrentSession == null)
            {
                Debug.LogWarning("[SessionManager] 이름 저장 실패 — 현재 세션 없음");
                return;
            }

            bool hasChar = !string.IsNullOrWhiteSpace(characterName);
            bool hasObj  = !string.IsNullOrWhiteSpace(objectName);
            if (!hasChar && !hasObj) return;   // 입력 없음 → 변경·전송 없음(기존 이름 유지)

            if (hasChar)
            {
                string name = characterName.Trim();
                CurrentSession.character_npc_name = name;                 // 가이드 대사·배치 캐릭터 이름표
                if (CurrentSession.assets?.character != null)
                    CurrentSession.assets.character.npc_name = name;      // 캐릭터 에셋 메타 일치
            }
            if (hasObj && CurrentSession.assets?.@object != null)
                CurrentSession.assets.@object.object_name = objectName.Trim();   // 오브제 이름표

            // Server 모드: 백엔드에 영구 저장(광장 '이전 창작물 이름' 표시용)
            if (dataSourceMode == DataSourceMode.Server && _apiClient != null)
                StartCoroutine(_apiClient.SaveNames(
                    CurrentSession.session_id,
                    CurrentSession.character_npc_name,
                    CurrentSession.assets?.@object?.object_name));
        }

        /// <summary>
        /// 현재 세션의 bubble_text/object_name 을 함께 갱신한다.
        /// Mock: 메모리에만 반영.
        /// Server: 메모리 반영 + PATCH /sessions/{id}/profile API 호출.
        /// </summary>
        public void SetBubbleText(string bubbleText)
        {
            if (CurrentSession == null)
            {
                Debug.LogWarning("[SessionManager] bubble_text 저장 실패 — 현재 세션 없음");
                return;
            }

            CurrentSession.bubble_text = bubbleText;
            CurrentSession.object_name = bubbleText;

            if (dataSourceMode == DataSourceMode.Server && _apiClient != null)
                StartCoroutine(_apiClient.UpdateBubbleText(CurrentSession.session_id, bubbleText));
        }

        // ════════════════════════════════════════════════════════════
        // 종료(체험 완료) — 씬 리로드로 초기화 후 대기화면 복귀 (A안)
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 관람객 종료 버튼(QuitButton) 진입점. 현재 세션을 '완료' 처리(큐를 다음으로 넘김)하고
        /// 씬을 리로드해 대기화면으로 복귀한다(다음 관람객).
        /// 리로드가 이전 세션의 스폰물·상태를 전부 초기화하고, 이어서 BeginSessionAcquisition이
        /// 다음 세션을 받는다(대기화면은 리로드된 새 씬에서 자동 표시).
        ///
        /// 모드별로 다른 부분은 "현재 세션 완료 + 큐 다음으로" 한 단계뿐이며, 이후 흐름은 Mock·Server 동일하다:
        ///   Mock  : mock_plaza에 창작물 기록 + MockSessionLoader.AdvanceToNext() (인덱스 +1)
        ///   Server: 대기 큐의 현재 세션을 history로 이동 (POST /unity-queue/advance)
        ///
        /// ※ '진짜 앱 종료'(전시 중 수정/재빌드용)는 OperatorQuit이 담당한다 — 그쪽은 advance를 하지
        ///   않으므로 재시작 시 같은 세션부터 이어진다. (advance는 오직 이 '완료' 경로에서만 일어난다)
        /// </summary>
        public void EndCurrentSession()
        {
            StartCoroutine(EndCurrentSessionRoutine());
        }

        private IEnumerator EndCurrentSessionRoutine()
        {
            // ── 모드별 마무리 (유일한 분기) — 둘 다 "현재 세션 완료 + 큐 다음으로" 의미 ──
            if (dataSourceMode == DataSourceMode.Mock)
            {
                LegoTwin.Plaza.PlazaManager.Instance?.SaveCurrentSessionToMockPlaza();  // 창작물을 광장에 기록
                MockSessionLoader.AdvanceToNext();                                       // 큐 다음으로 (Server advance 대응)
            }
            else if (_apiClient != null)
                yield return _apiClient.AdvanceQueue(_ => { });

            // ── 이후 공통 (Mock·Server 동일) ──
            // 다음 세션을 새로 받도록 상태 초기화. Server는 sessionId를 비워 큐 자동 감지로 진입한다.
            sessionId      = null;
            CurrentSession = null;

            ReloadScene();
            yield return null;          // 리로드된 씬의 Awake/Start 완료 대기 (WaitingScreen 표시·구독)
            BeginSessionAcquisition();  // 대기화면 상태에서 다음 세션 획득 시작
        }

        private static void ReloadScene() =>
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }
}
