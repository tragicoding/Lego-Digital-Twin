using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using LegoTwin.Core;
using LegoTwin.Data;

namespace LegoTwin.Network
{
    /// <summary>
    /// 백엔드 REST API 클라이언트.
    /// Unity 개발자는 SessionManager를 통해 간접 호출한다.
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

        /// <summary>GET /unity/sessions/{session_id} 로 Unity용 데이터를 가져온다.</summary>
        public IEnumerator FetchUnitySession(string sessionId, System.Action<UnitySessionResponse> onSuccess)
        {
            string url = ServerConfig.UnitySessionUrl(sessionId);
            using var req = UnityWebRequest.Get(url);
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] FetchUnitySession 실패: {req.error}");
                yield break;
            }

            var data = JsonUtility.FromJson<UnitySessionResponse>(req.downloadHandler.text);
            onSuccess?.Invoke(data);
        }

        /// <summary>GET /sessions/{session_id}/status polling.</summary>
        public IEnumerator PollStatus(string sessionId, System.Action<SessionStatusResponse> onTick)
        {
            while (true)
            {
                using var req = UnityWebRequest.Get(ServerConfig.StatusUrl(sessionId));
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    var data = JsonUtility.FromJson<SessionStatusResponse>(req.downloadHandler.text);
                    onTick?.Invoke(data);
                    if (data.ready_for_unity) yield break;
                }

                yield return new WaitForSeconds(3f);
            }
        }
    }
}
