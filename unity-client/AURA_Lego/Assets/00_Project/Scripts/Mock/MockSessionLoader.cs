using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Mock
{
    /// <summary>
    /// Resources/Mock/mock_session.json을 로드해 SessionData를 반환한다.
    /// Server Mode에서는 사용되지 않는다.
    /// </summary>
    public static class MockSessionLoader
    {
        private const string ResourcePath = "Mock/mock_session";

        public static SessionData Load()
        {
            var asset = Resources.Load<TextAsset>(ResourcePath);
            if (asset == null)
            {
                Debug.LogError("[MockSessionLoader] Resources/Mock/mock_session.json 을 찾을 수 없습니다.");
                return null;
            }

            var data = JsonUtility.FromJson<SessionData>(asset.text);
            Debug.Log($"[MockSessionLoader] 로드 완료 — character: {data.assets?.character?.npc_name}, object: {data.assets?.@object?.object_name}");
            return data;
        }
    }
}
