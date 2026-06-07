using System.Collections.Generic;
using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Mock
{
    /// <summary>
    /// Mock 세션 데이터를 로드한다.
    ///
    /// 우선순위:
    ///   1. Resources/Mock/mock_session_pool.json — 복수 세션 풀 (PlayerPrefs로 순환)
    ///   2. Resources/Mock/mock_session.json      — 단일 세션 (fallback)
    ///
    /// 풀 순환 방식:
    ///   앱 실행마다 PlayerPrefs("MockSessionIndex")를 1씩 증가시켜
    ///   sessions[index % count] 를 현재 관람객 세션으로 사용한다.
    /// </summary>
    public static class MockSessionLoader
    {
        private const string PoolResourcePath    = "Mock/mock_session_pool";
        private const string SingleResourcePath  = "Mock/mock_session";
        private const string PlayerPrefsIndexKey = "MockSessionIndex";

        [System.Serializable]
        private class SessionPool { public List<SessionData> sessions; }

        public static SessionData Load()
        {
            // ── 1. 풀 파일 시도 ────────────────────────────────────────
            var poolAsset = Resources.Load<TextAsset>(PoolResourcePath);
            if (poolAsset != null)
            {
                var pool = JsonUtility.FromJson<SessionPool>(poolAsset.text);
                if (pool?.sessions != null && pool.sessions.Count > 0)
                {
                    int idx     = PlayerPrefs.GetInt(PlayerPrefsIndexKey, 0);
                    var session = pool.sessions[idx % pool.sessions.Count];

                    PlayerPrefs.SetInt(PlayerPrefsIndexKey, idx + 1);
                    PlayerPrefs.Save();
                    return session;
                }
            }

            // ── 2. 단일 파일 fallback ──────────────────────────────────
            var asset = Resources.Load<TextAsset>(SingleResourcePath);
            if (asset == null)
            {
                Debug.LogError("[MockSessionLoader] mock_session.json 을 찾을 수 없습니다.");
                return null;
            }

            var data = JsonUtility.FromJson<SessionData>(asset.text);
            return data;
        }
    }
}
