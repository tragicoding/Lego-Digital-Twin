using System;
using System.Collections.Generic;

namespace LegoTwin.Data
{
    [Serializable]
    public class CharacterAssetData
    {
        public string asset_id;
        public string model_url;
        public string role;         // guide_npc
        public string npc_name;
        public Dictionary<string, AnimationInfo> animations;
    }

    [Serializable]
    public class AnimationInfo
    {
        public string key;            // walk / hello
        public string display_name;   // 걷기 / 인사_01
        public string unity_function; // animation_walk / animation_Hello
    }
}
