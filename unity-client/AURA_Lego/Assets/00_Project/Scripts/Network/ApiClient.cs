using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using LegoTwin.Core;
using LegoTwin.Data;

namespace LegoTwin.Network
{
    /// <summary>
    /// FastAPI 서버 REST API 클라이언트.
    /// Server Mode에서 SessionManager가 간접 호출한다.
    /// Mock Mode에서는 사용되지 않는다.
    /// </summary>
    public class ApiClient : MonoBehaviour
    {
        public static ApiClient Instance { get; private set; }

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        /// <summary>
        /// GET /unity/sessions/{session_id} — Unity용 완성 데이터 수신.
        /// </summary>
        public IEnumerator FetchUnitySession(string sessionId, System.Action<SessionData> onSuccess)
        {
            string url = ServerConfig.UnitySessionUrl(sessionId);
            Debug.Log($"[ApiClient] FetchUnitySession: {url}");

            using var req = UnityWebRequest.Get(url);
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] FetchUnitySession 실패: {req.error}");
                yield break;
            }

            var data = JsonUtility.FromJson<SessionData>(req.downloadHandler.text);
            onSuccess?.Invoke(data);
        }

        /// <summary>
        /// GET /sessions/{session_id}/status — ready_for_unity = true 될 때까지 polling (3초 간격).
        /// </summary>
        public IEnumerator PollUntilReady(string sessionId, System.Action onReady)
        {
            Debug.Log($"[ApiClient] 상태 polling 시작: {sessionId}");
            while (true)
            {
                using var req = UnityWebRequest.Get(ServerConfig.StatusUrl(sessionId));
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    var status = JsonUtility.FromJson<SessionStatusResponse>(req.downloadHandler.text);
                    if (status.ready_for_unity)
                    {
                        Debug.Log("[ApiClient] ready_for_unity = true");
                        onReady?.Invoke();
                        yield break;
                    }
                }

                yield return new WaitForSeconds(3f);
            }
        }
    }

    // polling 전용 경량 응답 구조
    [System.Serializable]
    public class SessionStatusResponse
    {
        public string session_id;
        public bool ready_for_unity;
    }
}
