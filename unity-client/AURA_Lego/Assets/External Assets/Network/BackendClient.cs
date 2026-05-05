using System;
using System.Collections;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// 백엔드 WebSocket 연결 관리.
/// ws://localhost:8000/ws/unity 에 연결해 메시지를 수신하고
/// 타입별로 이벤트를 발행한다.
/// </summary>
public class BackendClient : MonoBehaviour
{
    public static BackendClient Instance { get; private set; }

    [Header("Connection")]
    [SerializeField] private string _host = "localhost";
    [SerializeField] private int _port = 8000;
    [SerializeField] private float _reconnectDelay = 3f;

    public event Action<string> OnModelReady;      // filename
    public event Action<string, float> OnAtmosphere; // atmosphere, lux
    public event Action<float> OnFireworks;          // intensity

    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;
    private bool _running;

    private void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void Start() => StartCoroutine(ConnectLoop());

    private IEnumerator ConnectLoop()
    {
        _running = true;
        while (_running)
        {
            yield return ConnectAsync();
            if (_running)
            {
                Debug.Log($"[BackendClient] {_reconnectDelay}초 후 재연결...");
                yield return new WaitForSeconds(_reconnectDelay);
            }
        }
    }

    private IEnumerator ConnectAsync()
    {
        _cts = new CancellationTokenSource();
        _ws  = new ClientWebSocket();
        var uri = new Uri($"ws://{_host}:{_port}/ws/unity");

        Debug.Log($"[BackendClient] 연결 중: {uri}");
        var connectTask = _ws.ConnectAsync(uri, _cts.Token);
        yield return new WaitUntil(() => connectTask.IsCompleted);

        if (connectTask.IsFaulted || _ws.State != WebSocketState.Open)
        {
            Debug.LogWarning("[BackendClient] 연결 실패");
            yield break;
        }

        Debug.Log("[BackendClient] 연결 성공");
        var receiveTask = Task.Run(() => ReceiveLoop(_cts.Token));
        yield return new WaitUntil(() => receiveTask.IsCompleted);
    }

    private async Task ReceiveLoop(CancellationToken ct)
    {
        var buf = new byte[8192];
        var sb  = new StringBuilder();

        while (_ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
        {
            sb.Clear();
            WebSocketReceiveResult result;
            do
            {
                result = await _ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                sb.Append(Encoding.UTF8.GetString(buf, 0, result.Count));
            }
            while (!result.EndOfMessage);

            if (result.MessageType == WebSocketMessageType.Close) break;

            var json = sb.ToString();
            UnityMainThread.Enqueue(() => HandleMessage(json));
        }
    }

    private void HandleMessage(string json)
    {
        var msg = JsonUtility.FromJson<WsMessage>(json);
        if (msg == null) return;

        switch (msg.type)
        {
            case "model_ready":
                Debug.Log($"[BackendClient] model_ready: {msg.filename}");
                OnModelReady?.Invoke(msg.filename);
                break;

            case "atmosphere":
                Debug.Log($"[BackendClient] atmosphere: {msg.atmosphere} lux={msg.lux}");
                OnAtmosphere?.Invoke(msg.atmosphere, msg.lux);
                break;

            case "fireworks":
                Debug.Log($"[BackendClient] fireworks intensity={msg.intensity}");
                OnFireworks?.Invoke(msg.intensity);
                break;

            case "connected":
                Debug.Log("[BackendClient] 서버 연결 확인");
                break;
        }
    }

    private void OnDestroy()
    {
        _running = false;
        _cts?.Cancel();
        _ws?.Dispose();
    }

    [Serializable]
    private class WsMessage
    {
        public string type;
        public string filename;
        public string atmosphere;
        public float  lux;
        public float  intensity;
    }
}
