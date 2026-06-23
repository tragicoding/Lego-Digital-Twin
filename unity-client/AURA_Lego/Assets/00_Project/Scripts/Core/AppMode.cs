namespace LegoTwin.Core
{
    /// <summary>
    /// 앱 실행 입력/렌더 모드.
    /// 데이터 소스(Mock/Server)와 무관 — 오직 하드웨어(VR HMD 착용 여부)로만 결정된다.
    /// 따라서 Mock·Server 어느 모드에서도 동일하게 동작한다.
    /// </summary>
    public enum AppMode
    {
        /// <summary>키보드 + 마우스 (비VR 데스크톱).</summary>
        DesktopKBM,
        /// <summary>VR 헤드셋 (PC 테더드).</summary>
        VR,
    }
}
