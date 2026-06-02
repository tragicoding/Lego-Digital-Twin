using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// 광장 배치용 캐릭터 컨트롤러.
    /// 가이드 NPC와 별개로 오브제 옆에 배치된 캐릭터가 모션을 수행한다.
    ///
    /// 동작 흐름:
    ///   GuideNPCController (Step 5)
    ///     → PlayMotionFromPrompt("춤춰줘")
    ///     → MotionPromptParser.Parse()
    ///     → MixamoMotionLibrary.GetClip()
    ///     → CharacterAnimationController.PlayMotionClip()
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] 이 컴포넌트를 배치 캐릭터 Prefab(루트)에 추가
    ///   [ ] motionLibrary 필드에 MixamoMotionLibrary ScriptableObject 연결
    ///   [ ] Base Animator Controller에 "Motion" 스테이트 및 "motion" Trigger 추가
    ///       (CharacterAnimationController 주석 참고)
    /// </summary>
    public class PlacedCharacterController : MonoBehaviour
    {
        [Header("모션 라이브러리")]
        [Tooltip("Assets > Create > MINIVERSE > Mixamo Motion Library 로 생성한 에셋 연결")]
        public MixamoMotionLibrary motionLibrary;

        private CharacterAnimationController _animation;
        private Animator    _animator;
        private MotionType  _signatureMotionType = MotionType.Idle;

        private void Awake()
        {
            _animation = GetComponent<CharacterAnimationController>();
            if (_animation == null)
                _animation = GetComponentInChildren<CharacterAnimationController>();

            _animator = GetComponentInChildren<Animator>();

            // 가이드 모드용 배치 캐릭터: 프롬프트 입력 전까지 정지
            // 광장 배치 캐릭터: SetupForPlaza()에서 즉시 해제됨
            if (_animator != null)
                _animator.speed = 0f;
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 사용자 입력 문자열을 파싱해 해당하는 Mixamo 모션을 재생한다.
        /// GuideNPCController Step 5에서 호출.
        ///
        /// 사용 예:
        ///   placedChar.PlayMotionFromPrompt("춤춰줘");
        /// </summary>
        public void PlayMotionFromPrompt(string input)
        {
            if (motionLibrary == null)
            {
                Debug.LogWarning("[PlacedCharacterController] motionLibrary가 연결되지 않았습니다.");
                return;
            }

            // 1. 키워드 파싱 → MotionType
            MotionType motionType = MotionPromptParser.Parse(input);

            // 2. MotionType → AnimationClip
            AnimationClip clip = motionLibrary.GetClip(motionType);
            if (clip == null)
            {
                Debug.LogWarning($"[PlacedCharacterController] '{input}' → 클립 없음, 모션 생략");
                return;
            }

            // 3. 클립 교체 후 재생
            if (_animation == null)
            {
                Debug.LogWarning("[PlacedCharacterController] CharacterAnimationController를 찾을 수 없습니다.");
                return;
            }

            // 첫 프롬프트 입력 시 애니메이션 재개 (speed=0 → 1)
            if (_animator != null && _animator.speed == 0f)
                _animator.speed = 1f;

            Debug.Log($"[PlacedCharacterController] '{input}' → {motionType} → {clip.name} (loop)");
            _animation.PlayMotionClipLooping(clip, BuildResetCallback());
        }

        /// <summary>
        /// MotionType을 직접 지정해 재생한다.
        /// 테스트 또는 이벤트 기반 직접 호출 시 사용.
        /// </summary>
        public void PlayMotion(MotionType type)
        {
            if (motionLibrary == null)
            {
                Debug.LogWarning("[PlacedCharacterController] motionLibrary가 연결되지 않았습니다.");
                return;
            }

            var clip = motionLibrary.GetClip(type);
            if (clip == null) return;

            _animation?.PlayMotionClip(clip);
        }

        // ════════════════════════════════════════════════════════════
        // 광장 배치용 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 광장 배치 시 PlazaManager가 호출.
        /// motionLibrary와 시그니처 동작을 주입하고 애니메이션을 재개한다.
        /// </summary>
        public void SetupForPlaza(MixamoMotionLibrary lib, MotionType signatureMotion)
        {
            motionLibrary        = lib;
            _signatureMotionType = signatureMotion;
            if (_animator != null) _animator.speed = 1f;
        }

        /// <summary>플레이어 접근 시 시그니처 동작을 루프 재생한다.</summary>
        public void PlaySignatureMotion()
        {
            if (_signatureMotionType == MotionType.Idle || motionLibrary == null || _animation == null) return;

            var clip = motionLibrary.GetClip(_signatureMotionType);
            if (clip == null) return;

            if (_animator != null && _animator.speed == 0f) _animator.speed = 1f;
            _animation.PlayMotionClipLooping(clip, BuildResetCallback());
            Debug.Log($"[PlacedCharacterController] 시그니처 동작 재생: {_signatureMotionType}");
        }

        /// <summary>플레이어 이탈 시 시그니처 동작을 중단하고 idle로 복귀한다.</summary>
        public void StopSignatureMotion() => _animation?.StopMotionLoop();

        // ════════════════════════════════════════════════════════════
        // 내부 유틸
        // ════════════════════════════════════════════════════════════

        // 루프 재시작 시 호출될 위치 복원 콜백을 생성한다.
        // 현재 시점의 position/rotation을 캡처해 루프마다 원래 자리로 되돌린다.
        private System.Action BuildResetCallback()
        {
            var pos          = transform.position;
            var rot          = transform.rotation;
            var animLocalPos = _animator != null ? _animator.transform.localPosition : Vector3.zero;
            var animLocalRot = _animator != null ? _animator.transform.localRotation : Quaternion.identity;

            return () =>
            {
                transform.SetPositionAndRotation(pos, rot);
                if (_animator != null)
                    _animator.transform.SetLocalPositionAndRotation(animLocalPos, animLocalRot);
            };
        }
    }
}
