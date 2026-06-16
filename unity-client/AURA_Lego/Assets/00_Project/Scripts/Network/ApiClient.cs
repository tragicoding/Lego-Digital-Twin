using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using LegoTwin.Core;
using LegoTwin.Data;

namespace LegoTwin.Network
{
    /// <summary>
    /// FastAPI 서버 REST API 클라이언트.
    /// Server Mode에서만 사용. Mock Mode에서는 MockSessionLoader를 사용한다.
    ///
    /// 사용 예:
    ///   StartCoroutine(ApiClient.Instance.FetchUnitySession(id, data => ...));
    ///   StartCoroutine(ApiClient.Instance.FetchPlazaSessions(plaza => ...));
    ///   StartCoroutine(ApiClient.Instance.LikeSession(id, result => ...));
    ///   StartCoroutine(ApiClient.Instance.UpdateBubbleText(id, text, () => ...));
    /// </summary>
    public class ApiClient : MonoBehaviour
    {
        public static ApiClient Instance { get; private set; }

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }

            if (transform.parent != null)
            {
                // DontDestroyOnLoad는 루트 오브젝트에서만 안전하므로 먼저 부모에서 분리한다.
                transform.SetParent(null, true);
            }

            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        // ══════════════════════════════════════════════════════════════
        // 현재 세션 (가이드 모드)
        // ══════════════════════════════════════════════════════════════

        /// <summary>GET /unity/sessions/{id} — 현재 관람객 세션 데이터</summary>
        public IEnumerator FetchUnitySession(string sessionId, Action<SessionData> onSuccess)
        {
            using var req = UnityWebRequest.Get(ServerConfig.UnitySessionUrl(sessionId));
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] FetchUnitySession 실패: {req.error}");
                yield break;
            }
            onSuccess?.Invoke(JsonUtility.FromJson<SessionData>(req.downloadHandler.text));
        }

        /// <summary>GET /sessions/{id}/status — ready_for_unity = true 될 때까지 polling</summary>
        public IEnumerator PollUntilReady(string sessionId, Action onReady, float interval = 3f)
        {
            while (true)
            {
                using var req = UnityWebRequest.Get(ServerConfig.StatusUrl(sessionId));
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    var status = JsonUtility.FromJson<SessionStatusResponse>(req.downloadHandler.text);
                    if (status.ready_for_unity) { onReady?.Invoke(); yield break; }
                }
                yield return new WaitForSeconds(interval);
            }
        }

        // ══════════════════════════════════════════════════════════════
        // 광장 (자유 모드)
        // ══════════════════════════════════════════════════════════════

        /// <summary>GET /unity/plaza/sessions — 광장 전체 세션 목록</summary>
        public IEnumerator FetchPlazaSessions(Action<PlazaResponse> onSuccess)
        {
            using var req = UnityWebRequest.Get(ServerConfig.PlazaSessionsUrl());
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] FetchPlazaSessions 실패: {req.error}");
                yield break;
            }
            onSuccess?.Invoke(JsonUtility.FromJson<PlazaResponse>(req.downloadHandler.text));
        }

        // ══════════════════════════════════════════════════════════════
        // 좋아요
        // ══════════════════════════════════════════════════════════════

        /// <summary>POST /sessions/{id}/like — 좋아요 +1</summary>
        public IEnumerator LikeSession(string sessionId, Action<LikeResponse> onSuccess = null)
        {
            using var req = new UnityWebRequest(ServerConfig.LikeUrl(sessionId), "POST");
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] LikeSession 실패: {req.error}");
                yield break;
            }
            onSuccess?.Invoke(JsonUtility.FromJson<LikeResponse>(req.downloadHandler.text));
        }

        // ══════════════════════════════════════════════════════════════
        // 시그니처 동작 / 말풍선
        // ══════════════════════════════════════════════════════════════

        /// <summary>PATCH /sessions/{id}/signature-motion — 시그니처 동작 저장</summary>
        public IEnumerator SaveSignatureMotion(string sessionId, string motionType, Action onSuccess = null)
        {
            var payload = new SignatureMotionRequest { signature_motion = motionType };
            using var req = new UnityWebRequest(ServerConfig.SignatureMotionUrl(sessionId), "PATCH");
            req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(JsonUtility.ToJson(payload)));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"[ApiClient] SaveSignatureMotion 실패: {req.error}");
            else
                onSuccess?.Invoke();
        }

        /// <summary>
        /// PATCH /sessions/{id}/names — 캐릭터/오브제 이름 저장.
        /// 이름 입력은 Unity(대기화면)에서 받지만, 광장에서 '이전 창작물 이름'을 표시하려면 백엔드에
        /// 영구 저장돼야 하므로 전송한다. 백엔드는 이 값을 보관하고 GET /unity/plaza/sessions 응답의
        /// character_npc_name(top-level) · assets.object.object_name(nested)으로 돌려줘야 한다.
        /// </summary>
        public IEnumerator SaveNames(string sessionId, string characterName, string objectName, Action onSuccess = null)
        {
            var payload = new NamesRequest { character_npc_name = characterName, object_name = objectName };
            using var req = new UnityWebRequest(ServerConfig.NamesUrl(sessionId), "PATCH");
            req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(JsonUtility.ToJson(payload)));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"[ApiClient] SaveNames 실패: {req.error}");
            else
                onSuccess?.Invoke();
        }

        /// <summary>PATCH /sessions/{id}/profile — bubble_text 업데이트</summary>
        public IEnumerator UpdateBubbleText(string sessionId, string bubbleText, Action onSuccess = null)
        {
            var payload = new BubbleTextRequest { bubble_text = bubbleText };
            using var req = new UnityWebRequest(ServerConfig.BubbleTextUrl(sessionId), "PATCH");
            req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(JsonUtility.ToJson(payload)));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"[ApiClient] UpdateBubbleText 실패: {req.error}");
            else
                onSuccess?.Invoke();
        }

        // ══════════════════════════════════════════════════════════════
        // 대기 큐
        // ══════════════════════════════════════════════════════════════

        /// <summary>GET /sessions/active — 대기 큐 맨 앞 세션 ID (없으면 null)</summary>
        public IEnumerator FetchActiveSession(Action<string> onResult)
        {
            using var req = UnityWebRequest.Get(ServerConfig.ActiveSessionUrl());
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] FetchActiveSession 실패: {req.error}");
                onResult?.Invoke(null);
                yield break;
            }
            var resp = JsonUtility.FromJson<ActiveSessionResponse>(req.downloadHandler.text);
            onResult?.Invoke(resp?.session_id);
        }

        /// <summary>POST /sessions/unity-queue/advance — 현재 세션 제거 후 다음 세션 ID 반환 (없으면 null)</summary>
        public IEnumerator AdvanceQueue(Action<string> onResult)
        {
            using var req = new UnityWebRequest(ServerConfig.AdvanceQueueUrl(), "POST");
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[ApiClient] AdvanceQueue 실패: {req.error}");
                onResult?.Invoke(null);
                yield break;
            }
            var resp = JsonUtility.FromJson<AdvanceQueueResponse>(req.downloadHandler.text);
            onResult?.Invoke(resp?.next_session_id);
        }
    }

    [Serializable]
    internal class SessionStatusResponse
    {
        public string session_id;
        public bool   ready_for_unity;
    }

    [Serializable]
    internal class ActiveSessionResponse
    {
        public string session_id;
    }

    [Serializable]
    internal class AdvanceQueueResponse
    {
        public string removed;
        public string next_session_id;
    }
}
