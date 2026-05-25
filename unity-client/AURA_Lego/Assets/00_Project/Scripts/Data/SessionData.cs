using System;
using System.Collections.Generic;

namespace LegoTwin.Data
{
    // ════════════════════════════════════════════════════════════════
    // 서버 응답 / Mock JSON 공통 구조
    // GET /unity/sessions/{id}     → SessionData     (현재 세션, 가이드 모드)
    // GET /unity/plaza/sessions    → PlazaResponse   (광장 전체, 자유 모드)
    // mock_session.json            → 동일 구조 사용
    // ════════════════════════════════════════════════════════════════

    // ── 현재 관람객 세션 ─────────────────────────────────────────────

    [Serializable]
    public class SessionData
    {
        public string session_id;
        public string character_npc_name;
        public string object_name;
        public string bubble_text;
        public int    likes;
        public bool   ready_for_unity;
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
        public string model_url;      // FBX URL (서버) or "" (Mock → Inspector prefab 사용)
        public string texture_url;    // pbr_model GLB URL (텍스쳐 소스, glTFast로 로드)
        public string role;           // guide_npc
        public string npc_name;
    }

    // ── 오브제 ───────────────────────────────────────────────────────

    [Serializable]
    public class ObjectAssetData
    {
        public string asset_id;
        public string model_url;      // GLB URL (서버) or "" (Mock → Inspector prefab 사용)
        public string role;           // static_object
        public string object_name;
    }

    // ════════════════════════════════════════════════════════════════
    // 광장 (자유 모드) — GET /unity/plaza/sessions
    // ════════════════════════════════════════════════════════════════

    [Serializable]
    public class PlazaResponse
    {
        public List<PlazaSessionData> sessions;
        public string top_session_id;           // 현재 좋아요 1위 세션 ID
    }

    [Serializable]
    public class PlazaSessionData
    {
        public string session_id;
        public string character_npc_name;
        public string bubble_text;
        public int    likes;
        public bool   is_top_liked;             // true → 별 표시
        public SessionAssets assets;
    }

    // ════════════════════════════════════════════════════════════════
    // WebSocket 이벤트
    // ════════════════════════════════════════════════════════════════

    [Serializable]
    public class WsEvent
    {
        public string @event;         // "session_ready" | "likes_updated"
        public string session_id;
    }

    [Serializable]
    public class WsLikesEvent : WsEvent
    {
        public int    likes;
        public string top_session_id;
    }

    // ════════════════════════════════════════════════════════════════
    // 좋아요 API 응답
    // ════════════════════════════════════════════════════════════════

    [Serializable]
    public class LikeResponse
    {
        public string session_id;
        public int    likes;
        public bool   is_top_liked;
        public string top_session_id;
    }
}
