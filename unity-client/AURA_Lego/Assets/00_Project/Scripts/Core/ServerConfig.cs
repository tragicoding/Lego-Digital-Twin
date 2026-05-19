using UnityEngine;

namespace LegoTwin.Core
{
    public static class ServerConfig
    {
        public static string BaseUrl => "http://localhost:8000";
        public static string WsUrl  => "ws://localhost:8000/ws/unity";

        public static string SessionsUrl               => $"{BaseUrl}/sessions";
        public static string AssetsUrl(string sid)     => $"{BaseUrl}/sessions/{sid}/assets";
        public static string StatusUrl(string sid)     => $"{BaseUrl}/sessions/{sid}/status";
        public static string ProfileUrl(string sid)    => $"{BaseUrl}/sessions/{sid}/profile";
        public static string UnitySessionUrl(string sid) => $"{BaseUrl}/unity/sessions/{sid}";
        public static string ModelDownloadUrl(string filename) => $"{BaseUrl}/static/models/{filename}";
    }
}
