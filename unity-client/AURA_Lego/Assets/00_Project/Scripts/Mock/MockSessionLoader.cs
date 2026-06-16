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
    /// 큐(순환) 방식 — Server 모드와 동일한 의미:
    ///   PlayerPrefs("MockSessionIndex")가 '현재 큐 위치'를 가리킨다.
    ///   - Load()        : 현재 위치 세션을 반환만 한다(증가 X) → "큐 맨 앞 peek".
    ///                     (Server의 GET /sessions/active 에 대응 — 비파괴적 조회)
    ///   - AdvanceToNext(): 세션을 '완료'했을 때만 다음으로 넘긴다(증가).
    ///                     (Server의 POST /unity-queue/advance 에 대응)
    ///   → 완료 처리 없이 앱이 종료/재시작되면(운영자 종료·크래시) 같은 세션부터 다시 이어진다.
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
                    // 현재 큐 위치 세션을 반환만 한다(증가는 AdvanceToNext에서). → 재시작 시 같은 세션 이어짐.
                    int idx = PlayerPrefs.GetInt(PlayerPrefsIndexKey, 0);
                    return pool.sessions[idx % pool.sessions.Count];
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

        /// <summary>
        /// 현재 세션을 '완료'하고 큐를 다음 세션으로 넘긴다(인덱스 +1).
        /// SessionManager.EndCurrentSession(관람객 종료)에서만 호출 — Server의 POST /unity-queue/advance에 대응.
        /// 운영자 종료(OperatorQuit)는 이걸 호출하지 않으므로 재시작 시 같은 세션이 유지된다.
        /// </summary>
        public static void AdvanceToNext()
        {
            int idx = PlayerPrefs.GetInt(PlayerPrefsIndexKey, 0);
            PlayerPrefs.SetInt(PlayerPrefsIndexKey, idx + 1);
            PlayerPrefs.Save();
        }
    }
}
