using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using LegoTwin.Core;
using LegoTwin.Data;

namespace LegoTwin.Network
{
    /// <summary>
    /// 서버 WebSocket 연결 관리자 (ws://server/ws/unity).
    /// Server Mode에서 자동 실행. Mock Mode에서는 비활성.
    ///
    /// 수신 이벤트:
    ///   session_ready  → OnSessionReady  (가이드 모드 진입 트리거)
    ///   likes_updated  → OnLikesUpdated  (광장 좋아요 실시간 갱신)
    ///
    /// 사용 예:
    ///   WebSocketManager.Instance.OnLikesUpdated += HandleLikesUpdate;
    ///
    /// Unity WebGL/Android 에서는 NativeWebSocket 패키지를 사용 권장.
    /// 현재는 UnityWebRequest Long-Polling 방식으로 구현 (WebSocket 패키지 미포함 시 대체).
    /// TODO: NativeWebSocket 패키지 임포트 후 아래 주석 참고해서 교체
    /// </summary>
    public class WebSocketManager : MonoBehaviour
    {
        public static WebSocketManager Instance { get; private set; }

        /// <summary>세션 준비 완료 이벤트 (session_id)</summary>
        public event Action<string> OnSessionReady;

        /// <summary>좋아요 업데이트 이벤트</summary>
        public event Action<WsLikesEvent> OnLikesUpdated;

        [Header("재연결 설정")]
        [Tooltip("폴링 간격 (초) — NativeWebSocket 사용 시 무시")]
        public float pollInterval = 2f;

        private bool _running;

        // ── TODO: NativeWebSocket 패키지로 교체 시 아래 구조 사용 ────
        // using NativeWebSocket;
        // private WebSocket _ws;
        //
        // async void ConnectWs() {
        //     _ws = new WebSocket(ServerConfig.WsUrl);
        //     _ws.OnMessage += bytes => HandleMessage(System.Text.Encoding.UTF8.GetString(bytes));
        //     await _ws.Connect();
        // }
        // void Update() { _ws?.DispatchMessageQueue(); }
        // ────────────────────────────────────────────────────────────

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        /// <summary>
        /// WebSocket 수신 시작. SessionManager.Server Mode에서 자동 호출.
        /// </summary>
        public void StartListening()
        {
            if (_running) return;
            _running = true;
            StartCoroutine(PollLoop());
            Debug.Log("[WebSocketManager] 수신 시작 (폴링 방식)");
        }

        public void StopListening()
        {
            _running = false;
            StopAllCoroutines();
        }

        // ── 폴링 루프 (NativeWebSocket 미사용 시 대체) ───────────────
        // 서버 /ws/unity 는 WebSocket 전용이므로 실제 연결 불가.
        // NativeWebSocket 적용 전까지는 HTTP polling으로 plaza를 주기적으로 갱신.
        // TODO: NativeWebSocket 패키지 적용 후 이 코루틴 제거하고 위 ConnectWs() 사용.

        private IEnumerator PollLoop()
        {
            while (_running)
            {
                // 광장 좋아요 갱신 — 폴링 방식으로 PlazaManager에 알림
                OnLikesUpdated?.Invoke(new WsLikesEvent { @event = "poll_tick" });
                yield return new WaitForSeconds(pollInterval);
            }
        }

        /// <summary>WebSocket 메시지 파싱 (NativeWebSocket 사용 시 여기 연결)</summary>
        public void HandleMessage(string json)
        {
            var ev = JsonUtility.FromJson<WsEvent>(json);
            switch (ev.@event)
            {
                case "session_ready":
                    Debug.Log($"[WebSocketManager] session_ready: {ev.session_id}");
                    OnSessionReady?.Invoke(ev.session_id);
                    break;

                case "likes_updated":
                    var likesEv = JsonUtility.FromJson<WsLikesEvent>(json);
                    Debug.Log($"[WebSocketManager] likes_updated: {likesEv.session_id} → {likesEv.likes}");
                    OnLikesUpdated?.Invoke(likesEv);
                    break;

                default:
                    Debug.LogWarning($"[WebSocketManager] 알 수 없는 이벤트: {ev.@event}");
                    break;
            }
        }
    }
}
