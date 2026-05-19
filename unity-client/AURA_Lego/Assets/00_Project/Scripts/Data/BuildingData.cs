using System;

namespace LegoTwin.Data
{
    [Serializable]
    public class BuildingAssetData
    {
        public string asset_id;
        public string model_url;
        public string role;        // static_building
        public string owner_name;
    }
}
