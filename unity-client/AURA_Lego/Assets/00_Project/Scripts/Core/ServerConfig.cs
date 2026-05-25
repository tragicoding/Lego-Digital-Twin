using System.IO;
using UnityEngine;

namespace LegoTwin.Core
{
    /// <summary>
    /// 서버 주소 설정.
    /// StreamingAssets/Config/server_config.json이 존재하면 그 값을 사용한다.
    /// 없으면 기본값 "http://localhost:8000" 사용 (개발 환경).
    ///
    /// 전시 환경:
    ///   StreamingAssets/Config/server_config.json 에 아래 내용 작성
    ///   { "base_url": "http://192.168.x.x:8000" }
    /// </summary>
    public static class ServerConfig
    {
        private static string _baseUrl;

        public static string BaseUrl
        {
            get
            {
                if (_baseUrl == null) _baseUrl = LoadBaseUrl();
                return _baseUrl;
            }
        }

        public static string WsUrl => BaseUrl
            .Replace("http://", "ws://")
            .Replace("https://", "wss://") + "/ws/unity";

        // ── 세션 ──────────────────────────────────────────────────────
        public static string UnitySessionUrl(string sid)  => $"{BaseUrl}/unity/sessions/{sid}";
        public static string StatusUrl(string sid)        => $"{BaseUrl}/sessions/{sid}/status";
        public static string LikeUrl(string sid)          => $"{BaseUrl}/sessions/{sid}/like";
        public static string BubbleTextUrl(string sid)    => $"{BaseUrl}/sessions/{sid}/profile";

        // ── 광장 ──────────────────────────────────────────────────────
        public static string PlazaSessionsUrl()           => $"{BaseUrl}/unity/plaza/sessions";

        // ── 모델 다운로드 ─────────────────────────────────────────────
        public static string ModelDownloadUrl(string url)  => url; // model_url은 이미 전체 URL

        private static string LoadBaseUrl()
        {
            string configPath = Path.Combine(Application.streamingAssetsPath, "Config", "server_config.json");
            if (File.Exists(configPath))
            {
                try
                {
                    var cfg = JsonUtility.FromJson<ServerConfigFile>(File.ReadAllText(configPath));
                    if (!string.IsNullOrEmpty(cfg.base_url))
                    {
                        Debug.Log($"[ServerConfig] 서버 주소 로드: {cfg.base_url}");
                        return cfg.base_url;
                    }
                }
                catch { /* 파싱 실패 시 기본값 사용 */ }
            }
            return "http://localhost:8000";
        }

        [System.Serializable]
        private class ServerConfigFile
        {
            public string base_url;
        }
    }
}
