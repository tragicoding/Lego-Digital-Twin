using System;

namespace LegoTwin.Data
{
    [Serializable]
    public class VehicleAssetData
    {
        public string asset_id;
        public string model_url;
        public string role;         // driveable_vehicle
        public string owner_name;
        public string control_mode; // auto_drive / manual
    }
}
