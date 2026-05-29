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
        private Animator _animator;

        private void Awake()
        {
            _animation = GetComponent<CharacterAnimationController>();
            if (_animation == null)
                _animation = GetComponentInChildren<CharacterAnimationController>();

            _animator = GetComponentInChildren<Animator>();

            // 프롬프트 입력 전까지 애니메이션 비활성
            if (_animator != null)
                _animator.enabled = false;
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

            // 첫 프롬프트 입력 시 Animator 활성화
            if (_animator != null && !_animator.enabled)
                _animator.enabled = true;

            Debug.Log($"[PlacedCharacterController] '{input}' → {motionType} → {clip.name} (loop)");
            _animation.PlayMotionClipLooping(clip);
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
    }
}
