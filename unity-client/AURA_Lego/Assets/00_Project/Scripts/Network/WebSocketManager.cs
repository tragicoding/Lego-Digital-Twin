using System;
using System.Text;
using UnityEngine;
using NativeWebSocket;
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
    ///   WebSocketManager.Instance.OnSessionReady  += sid => StartGuide(sid);
    ///   WebSocketManager.Instance.OnLikesUpdated  += ev  => plaza.ApplyLikes(ev);
    /// </summary>
    public class WebSocketManager : MonoBehaviour
    {
        public static WebSocketManager Instance { get; private set; }

        /// <summary>세션 준비 완료 이벤트 (session_id)</summary>
        public event Action<string>       OnSessionReady;

        /// <summary>좋아요 업데이트 이벤트</summary>
        public event Action<WsLikesEvent> OnLikesUpdated;

        [Header("재연결 설정")]
        [Tooltip("연결 끊김 후 재시도 대기 시간 (초)")]
        public float reconnectDelay = 3f;

        private WebSocket _ws;
        private bool      _running;
        private bool      _reconnecting;

        // ─────────────────────────────────────────────────────────────
        // Unity 생명주기
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            if (Instance != null) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        private void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            // NativeWebSocket: 메인스레드에서 메시지 큐 디스패치 필수
            _ws?.DispatchMessageQueue();
#endif
        }

        private async void OnApplicationQuit()
        {
            _running = false;
            if (_ws != null && _ws.State == WebSocketState.Open)
                await _ws.Close();
        }

        // ─────────────────────────────────────────────────────────────
        // 공개 API
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// WebSocket 수신 시작. SessionManager (Server Mode) 에서 자동 호출.
        /// </summary>
        public void StartListening()
        {
            if (_running) return;
            _running = true;
            ConnectWs();
            Debug.Log("[WebSocketManager] 연결 시작");
        }

        public async void StopListening()
        {
            _running = false;
            _reconnecting = false;
            if (_ws != null && _ws.State == WebSocketState.Open)
                await _ws.Close();
        }

        // ─────────────────────────────────────────────────────────────
        // 내부 연결 로직
        // ─────────────────────────────────────────────────────────────

        private async void ConnectWs()
        {
            if (!_running) return;

            string url = ServerConfig.WsUrl;
            Debug.Log($"[WebSocketManager] 연결 중: {url}");

            _ws = new WebSocket(url);

            _ws.OnOpen    += OnOpen;
            _ws.OnMessage += OnMessage;
            _ws.OnError   += OnError;
            _ws.OnClose   += OnClose;

            await _ws.Connect();
        }

        private void OnOpen()
        {
            _reconnecting = false;
            Debug.Log("[WebSocketManager] 연결 성공");
        }

        private void OnMessage(byte[] bytes)
        {
            HandleMessage(Encoding.UTF8.GetString(bytes));
        }

        private void OnError(string errorMsg)
        {
            Debug.LogWarning($"[WebSocketManager] 오류: {errorMsg}");
        }

        private void OnClose(WebSocketCloseCode code)
        {
            Debug.Log($"[WebSocketManager] 연결 종료 (코드: {code})");
            if (_running && !_reconnecting)
                ScheduleReconnect();
        }

        private void ScheduleReconnect()
        {
            _reconnecting = true;
            Debug.Log($"[WebSocketManager] {reconnectDelay}초 후 재연결 시도...");
            Invoke(nameof(Reconnect), reconnectDelay);
        }

        private void Reconnect()
        {
            if (!_running) return;
            ConnectWs();
        }

        // ─────────────────────────────────────────────────────────────
        // 메시지 파싱
        // ─────────────────────────────────────────────────────────────

        /// <summary>수신 JSON 파싱 후 이벤트 발행.</summary>
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
