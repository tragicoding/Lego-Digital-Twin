namespace LegoTwin.Character
{
    /// <summary>
    /// Mixamo 모션 종류 열거형.
    /// MotionPromptParser가 사용자 입력 문자열을 이 타입으로 변환하고,
    /// MixamoMotionLibrary가 이 타입에 해당하는 AnimationClip을 반환한다.
    ///
    /// 새 모션 추가 방법:
    ///   1. 이 enum에 항목 추가
    ///   2. MotionPromptParser의 _keywordMap에 키워드 추가
    ///   3. MixamoMotionLibrary ScriptableObject에 AnimationClip 연결
    ///   4. Assets/00_Project/Animations/Mixamo/ 폴더에 .anim 파일 추가
    /// </summary>
    public enum MotionType
    {
        Idle,       // 기본 대기 (매칭 없을 때 폴백)
        Dance,      // 춤, 댄스
        Run,        // 달려, 뛰어
        Jump,       // 점프, 뛰기
        Wave,       // 손흔들어, 인사
        Sit,        // 앉아
        Kick,       // 발차기, 킥
        Spin,       // 돌아, 회전
        Cheer,      // 만세, 환호
        Clap,       // 박수
        Victory,    // 승리, 파이팅
        Cry,        // 울어, 눈물, 슬퍼
        Walk,       // 걸어, 걷기, 산책
        Handstand,  // 물구나무, 핸드스탠드, 헤드스핀
        Hiphop,     // 힙합, 비보잉
    }
}
