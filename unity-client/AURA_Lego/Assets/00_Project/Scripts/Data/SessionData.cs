using System;
using System.Collections.Generic;

namespace LegoTwin.Data
{
    [Serializable]
    public class SessionStatusResponse
    {
        public string session_id;
        public bool profile_completed;
        public Dictionary<string, AssetStatusItem> assets;
        public bool ready_for_unity;
    }

    [Serializable]
    public class AssetStatusItem
    {
        public string status;   // queued / processing / completed / failed
        public string stage;    // waiting / generating / rigging / animating / downloading / ready
        public int progress;
        public string model_url;
    }

    [Serializable]
    public class UnitySessionResponse
    {
        public string session_id;
        public string nickname;
        public string bubble_text;
        public bool ready_for_unity;
        public UnityAssets assets;
    }

    [Serializable]
    public class UnityAssets
    {
        public CharacterAssetData character;
        public BuildingAssetData building;
        public VehicleAssetData vehicle;
    }
}
