using System.Collections.Generic;
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
    ///     → MixamoMotionLibrary.GetRandomClip(직전 클립 제외)
    ///     → CharacterAnimationController.PlayMotionClipLooping()
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
        private Animator      _animator;

        // 시그니처 동작 = 관람객이 저장 시점에 고른 "특정 클립" (타입이 아니라 클립 자체).
        // 광장에서 다가올 때마다 항상 이 클립 그대로 재생된다.
        private AnimationClip _signatureClip;

        // 동작 타입별 직전 재생 클립 기억 — 같은 동작 반복 입력 시 직전과 다른 변형을 고르기 위함.
        // 캐릭터 인스턴스마다 독립 (라이브러리 에셋은 상태를 갖지 않음).
        private readonly Dictionary<MotionType, AnimationClip> _lastClipByType
            = new Dictionary<MotionType, AnimationClip>();

        // 가장 최근 프롬프트로 실제 재생된 클립 — 시그니처 저장 시 이 클립이 그대로 저장된다.
        private AnimationClip _lastPromptClip;

        /// <summary>가장 최근 프롬프트로 재생된 클립의 이름(없으면 null). 시그니처 저장용.</summary>
        public string LastPromptClipName => _lastPromptClip != null ? _lastPromptClip.name : null;

        // 초기(스폰) 자세 — 모션 루프의 Root Motion 누적 방지용. Awake에서 한 번만 캡처.
        private Vector3    _initialPos;
        private Quaternion _initialRot;
        private Vector3    _animInitialLocalPos;
        private Quaternion _animInitialLocalRot;

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

            // 스폰 직후 자세를 고정 기준으로 저장 (이후 모든 모션이 이 자리로 복원됨)
            _initialPos = transform.position;
            _initialRot = transform.rotation;
            if (_animator != null)
            {
                _animInitialLocalPos = _animator.transform.localPosition;
                _animInitialLocalRot = _animator.transform.localRotation;
            }
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 사용자 입력 문자열을 파싱해 해당하는 Mixamo 모션을 재생한다.
        /// GuideNPCController Step 5에서 호출.
        ///
        /// 반환값: 입력이 알려진 동작 키워드와 매칭됐으면 true,
        ///         매칭되는 동작이 없어 Cry 로 폴백 재생했으면 false.
        ///         (호출자가 미인식 입력을 구분해 재입력을 유도하는 데 사용)
        ///
        /// 사용 예:
        ///   bool recognized = placedChar.PlayMotionFromPrompt("춤춰줘");
        /// </summary>
        public bool PlayMotionFromPrompt(string input)
        {
            if (motionLibrary == null)
            {
                Debug.LogWarning("[PlacedCharacterController] motionLibrary가 연결되지 않았습니다.");
                return true;   // 설정 오류 — 재입력 유도(false) 대상이 아니므로 흐름은 진행
            }

            // 1. 키워드 파싱 → MotionType (인식되는 동작이 없으면 Idle 반환)
            MotionType motionType = MotionPromptParser.Parse(input);

            // 입력이 알려진 동작과 매칭됐는지 (Idle = 매칭 키워드 없음 = 미인식)
            bool recognized = motionType != MotionType.Idle;

            // 입력은 했지만 매칭되는 동작이 없으면 Cry 로 대체.
            // (빈 입력은 제외 — Cry 로 가지 않음)
            if (!recognized && !string.IsNullOrWhiteSpace(input))
                motionType = MotionType.Cry;

            // 2. MotionType → AnimationClip (직전 재생 클립 제외, 같은 동작이라도 다른 변형 재생)
            _lastClipByType.TryGetValue(motionType, out var lastClip);
            AnimationClip clip = motionLibrary.GetRandomClip(motionType, lastClip);
            if (clip == null)
            {
                Debug.LogWarning($"[PlacedCharacterController] '{input}' → 클립 없음, 모션 생략");
                return recognized;
            }
            _lastClipByType[motionType] = clip;
            _lastPromptClip = clip;   // 시그니처 저장 시 이 클립이 그대로 저장됨

            // 3. 클립 교체 후 재생
            if (_animation == null)
            {
                Debug.LogWarning("[PlacedCharacterController] CharacterAnimationController를 찾을 수 없습니다.");
                return recognized;
            }

            // 첫 프롬프트 입력 시 애니메이션 재개 (speed=0 → 1)
            if (_animator != null && _animator.speed == 0f)
                _animator.speed = 1f;

            // 새 모션 시작 전 초기 자리로 복원 (이전 모션 드리프트 제거)
            ResetToInitialPose();
            _animation.PlayMotionClipLooping(clip, ResetToInitialPose);

            return recognized;
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
        /// motionLibrary와 시그니처 클립(관람객이 고른 특정 클립)을 주입하고 애니메이션을 재개한다.
        /// signatureClip 이 null 이면 시그니처 미설정 — 접근해도 재생하지 않는다.
        /// </summary>
        public void SetupForPlaza(MixamoMotionLibrary lib, AnimationClip signatureClip)
        {
            motionLibrary  = lib;
            _signatureClip = signatureClip;
            if (_animator != null) _animator.speed = 1f;
        }

        /// <summary>
        /// 플레이어 접근 시 시그니처 동작을 루프 재생한다.
        /// 관람객이 저장한 그 클립을 항상 동일하게 재생한다(접근할 때마다 같은 동작).
        /// </summary>
        public void PlaySignatureMotion()
        {
            if (_signatureClip == null || _animation == null) return;

            if (_animator != null && _animator.speed == 0f) _animator.speed = 1f;

            ResetToInitialPose();
            _animation.PlayMotionClipLooping(_signatureClip, ResetToInitialPose);
        }

        /// <summary>플레이어 이탈 시 시그니처 동작을 중단하고 idle로 복귀한다.</summary>
        public void StopSignatureMotion() => _animation?.StopMotionLoop();

        /// <summary>
        /// 드리프트 보정 기준(앵커)을 현재 transform 자세로 다시 잡는다.
        /// 배회 캐릭터처럼 스폰 위치에서 이동한 뒤 '멈춰 선 그 자리'에서 시그니처를 재생할 때,
        /// PlaySignatureMotion 의 초기 자세 복원이 스폰 위치가 아니라 현재 위치를 기준으로
        /// 동작하도록 한다. (호출 직후 PlaySignatureMotion 을 호출)
        /// 정지 배치 캐릭터는 호출하지 않으면 기존과 100% 동일하게 스폰 위치를 유지한다.
        /// </summary>
        public void RebaseAnchorToCurrent()
        {
            _initialPos = transform.position;
            _initialRot = transform.rotation;
            if (_animator != null)
            {
                _animInitialLocalPos = _animator.transform.localPosition;
                _animInitialLocalRot = _animator.transform.localRotation;
            }
        }

        // ════════════════════════════════════════════════════════════
        // 내부 유틸
        // ════════════════════════════════════════════════════════════

        // 고정된 초기(스폰) 자세로 복원한다. 모션 루프 재시작·새 모션 시작마다 호출되어
        // Root Motion 누적으로 위치/회전이 드리프트하는 것을 막는다.
        private void ResetToInitialPose()
        {
            transform.SetPositionAndRotation(_initialPos, _initialRot);
            if (_animator != null)
                _animator.transform.SetLocalPositionAndRotation(_animInitialLocalPos, _animInitialLocalRot);
        }
    }
}
