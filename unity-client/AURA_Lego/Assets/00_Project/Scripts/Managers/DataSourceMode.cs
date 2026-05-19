namespace LegoTwin.Managers
{
    public enum DataSourceMode
    {
        /// <summary>
        /// 서버 없이 Resources/Mock/mock_session.json을 사용.
        /// 팀원이 로컬에서 시나리오/애니메이션/씬 개발 시 사용.
        /// </summary>
        Mock,

        /// <summary>
        /// FastAPI 서버에서 실제 세션 데이터를 수신.
        /// 전시 통합 테스트 및 실제 전시 시 사용.
        /// </summary>
        Server
    }
}
