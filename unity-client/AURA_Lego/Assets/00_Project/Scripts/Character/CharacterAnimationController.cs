using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 캐릭터 애니메이션 컨트롤러.
    /// 캐릭터 루트 GameObject에 추가한다.
    ///
    /// 팀원 사용 예:
    ///   npc.Animation.animation_walk();
    ///   npc.Animation.animation_idle();
    ///   npc.Animation.PlayAnimation("walk");
    ///   npc.Animation.PlayMotionClip(clip);   // Mixamo 클립 런타임 교체
    /// </summary>
    public class CharacterAnimationController : MonoBehaviour
    {
        [Header("NPC Info")]
        public string npcName;
        public string bubbleText;

        [Header("Animator 파라미터")]
        [Tooltip("Bool 방식 파라미터 이름 (Animator에 이 Bool이 있으면 우선 사용)")]
        public string walkBoolParam = "isWalking";
        [Tooltip("Trigger 방식 — 걷기 Trigger 이름 (Bool 없을 때 사용)")]
        public string walkTrigger   = "walk";
        [Tooltip("Trigger 방식 — 정지 Trigger 이름 (Bool 없을 때 사용)")]
        public string idleTrigger   = "idle";

        private Animator _animator;
        private AnimatorOverrideController _overrideController;
        private bool _useBoolParam;

        // AnimatorOverrideController에서 교체할 클립 슬롯 이름
        // Base Animator Controller의 "Motion" 스테이트 클립 이름과 일치해야 함
        private const string MOTION_SLOT = "Motion";

        private void Awake()
        {
            _animator = GetComponentInChildren<Animator>();
            if (_animator == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] Animator 없음: {gameObject.name}");
                return;
            }
            if (_animator.runtimeAnimatorController == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] Animator Controller 미연결: {gameObject.name}\n" +
                                 "Inspector에서 Animator Controller를 연결해야 애니메이션이 재생됩니다.\n" +
                                 "물리적 이동(transform)은 Controller 없이도 동작합니다.");
                return;
            }
            InitOverrideController();
            DetectAnimatorMode();
        }

        /// <summary>
        /// AnimatorOverrideController 초기화.
        /// Base Animator Controller가 설정되어 있어야 동작한다.
        /// </summary>
        private void InitOverrideController()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null)
                return;

            // 이미 Override라면 중복 생성 방지
            if (_animator.runtimeAnimatorController is AnimatorOverrideController)
            {
                _overrideController = _animator.runtimeAnimatorController as AnimatorOverrideController;
                return;
            }

            _overrideController = new AnimatorOverrideController(_animator.runtimeAnimatorController);
            _animator.runtimeAnimatorController = _overrideController;
        }

        /// <summary>
        /// Animator 파라미터를 확인하여 Bool / Trigger 방식을 자동 감지합니다.
        /// </summary>
        private void DetectAnimatorMode()
        {
            foreach (var p in _animator.parameters)
            {
                if (p.name == walkBoolParam && p.type == AnimatorControllerParameterType.Bool)
                {
                    _useBoolParam = true;
                    Debug.Log($"[CharacterAnimationController] {gameObject.name}: Bool 방식 ('{walkBoolParam}')");
                    return;
                }
            }
            Debug.Log($"[CharacterAnimationController] {gameObject.name}: Trigger 방식 ('{walkTrigger}'/'{idleTrigger}')");
        }

        // ── 공개 애니메이션 함수 ─────────────────────────────────────

        /// <summary>걷기 애니메이션 (Bool / Trigger 자동 분기)</summary>
        public void animation_walk()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            if (_useBoolParam) _animator.SetBool(walkBoolParam, true);
            else               PlayAnimation(walkTrigger);
        }

        /// <summary>idle(대기) 애니메이션 (Bool / Trigger 자동 분기)</summary>
        public void animation_idle()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            if (_useBoolParam) _animator.SetBool(walkBoolParam, false);
            else               PlayAnimation(idleTrigger);
        }

        /// <summary>animation key 기반 Trigger 실행</summary>
        public void PlayAnimation(string animationKey)
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            _animator.SetTrigger(animationKey);
            Debug.Log($"[CharacterAnimationController] {npcName}: {animationKey}");
        }

        /// <summary>
        /// Mixamo AnimationClip을 런타임에 교체하여 재생한다.
        /// AnimatorOverrideController의 "Motion" 슬롯을 교체한 뒤
        /// "motion" Trigger를 발생시킨다.
        ///
        /// 유니티 개발자 체크리스트:
        ///   [ ] Base Animator Controller에 "Motion" 스테이트 추가
        ///       (Any State → Motion, Trigger 파라미터: "motion")
        /// </summary>
        public void PlayMotionClip(AnimationClip clip)
        {
            if (clip == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] clip이 null입니다.");
                return;
            }

            if (_overrideController == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] OverrideController 없음 — " +
                                 $"Base Animator Controller가 설정되어 있는지 확인하세요.");
                return;
            }

            _overrideController[MOTION_SLOT] = clip;
            _animator.SetTrigger("motion");
            Debug.Log($"[CharacterAnimationController] {npcName}: Motion → {clip.name}");
        }

        // ── 초기화 ───────────────────────────────────────────────────

        /// <summary>CharacterAssetData로 NPC 정보를 초기화한다.</summary>
        public void Initialize(CharacterAssetData data, string bubble = "")
        {
            if (data == null) return;
            npcName    = data.npc_name;
            bubbleText = bubble;

            // ── Mock Mode ──────────────────────────────────────────────────────
            // GeneratedCharacters/**/character/ 폴더의 *_rigged.fbx + *_texture.glb 쌍은
            // Editor 스크립트 (CharacterTextureApplier) 가 임포트 시 자동으로 텍스쳐를 적용합니다.
            // Prefab을 GeneratedCharacterSpawner.mockCharacterPrefab 에 연결하면 됩니다.
            //
            // ── Server Mode ────────────────────────────────────────────────────
            // data.model_url   → 리깅된 FBX URL (TriLib 런타임 로드 필요)
            // data.texture_url → PBR 텍스쳐 GLB URL (glTFast로 머티리얼 추출 후 FBX에 적용)
            // Server Mode 텍스쳐 URL 이 있을 때:
            // → 위 TODO 구현 완료 후 ApplyTextureFromUrl(data.texture_url) 호출 추가
        }

        // ── Server Mode — GLB URL 에서 텍스쳐 추출 후 적용 ─────────────────
        //
        // TODO: 유니티 개발자 — 아래 주석을 실제 구현으로 교체하세요.
        //
        // glTFast 6.x 비동기 로드 예시:
        //
        //   private async void ApplyTextureFromUrl(string url)
        //   {
        //       var gltf    = new GLTFast.GltfImport();
        //       bool success = await gltf.Load(url);
        //       if (!success) { Debug.LogWarning($"GLB 로드 실패: {url}"); return; }
        //
        //       var renderer = GetComponentInChildren<SkinnedMeshRenderer>();
        //       if (renderer != null)
        //       {
        //           var mat = gltf.GetMaterial(0);
        //           if (mat != null) renderer.material = mat;
        //       }
        //   }
        //
        // Initialize() 에서 호출:
        //   if (!string.IsNullOrEmpty(data.texture_url))
        //       ApplyTextureFromUrl(data.texture_url);
    }
}
