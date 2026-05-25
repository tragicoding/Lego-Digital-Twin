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
        // 말풍선 텍스트 업데이트 (가이드 모드 → 자유 모드 전환 전)
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// PATCH /sessions/{id}/profile — bubble_text 업데이트.
        /// 가이드 시나리오 마지막 단계에서 관람객이 인사말 입력 후 호출.
        /// 자유 모드 진입 시 광장 캐릭터의 말풍선에 자동 반영.
        /// </summary>
        public IEnumerator UpdateBubbleText(string sessionId, string bubbleText, Action onSuccess = null)
        {
            string body = $"{{\"bubble_text\":\"{bubbleText}\"}}";
            using var req = new UnityWebRequest(ServerConfig.BubbleTextUrl(sessionId), "PATCH");
            req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"[ApiClient] UpdateBubbleText 실패: {req.error}");
            else
                onSuccess?.Invoke();
        }
    }

    [Serializable]
    internal class SessionStatusResponse
    {
        public string session_id;
        public bool   ready_for_unity;
    }
}
