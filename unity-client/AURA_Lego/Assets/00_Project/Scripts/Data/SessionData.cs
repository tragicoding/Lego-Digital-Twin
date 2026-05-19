using System;

namespace LegoTwin.Data
{
    // ── 서버 응답 / Mock JSON 공통 구조 ─────────────────────────────
    // GET /unity/sessions/{id} 응답과 mock_session.json이 동일한 구조를 사용한다.

    [Serializable]
    public class SessionData
    {
        public string session_id;
        public string character_npc_name;
        public string object_name;
        public string bubble_text;
        public bool ready_for_unity;
        public SessionAssets assets;
    }

    [Serializable]
    public class SessionAssets
    {
        public CharacterAssetData character;

        // "object"는 C# 예약어 → @ 접두어 사용, JSON 키 "object"와 자동 매핑
        public ObjectAssetData @object;
    }

    // ── 캐릭터 ───────────────────────────────────────────────────────
    [Serializable]
    public class CharacterAssetData
    {
        public string asset_id;
        public string model_url;
        public string role;           // guide_npc
        public string npc_name;
        public AnimationMap animations;
    }

    [Serializable]
    public class AnimationMap
    {
        public AnimationInfo walk;
        public AnimationInfo idle;
    }

    [Serializable]
    public class AnimationInfo
    {
        public string key;            // walk / idle
        public string display_name;
        public string unity_function; // animation_walk / animation_idle
    }

    // ── 오브제 ───────────────────────────────────────────────────────
    [Serializable]
    public class ObjectAssetData
    {
        public string asset_id;
        public string model_url;
        public string role;           // static_object
        public string object_name;
    }
}
