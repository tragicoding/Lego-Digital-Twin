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
    /// </summary>
    public class CharacterAnimationController : MonoBehaviour
    {
        [Header("NPC Info")]
        public string npcName;
        public string bubbleText;

        private Animator _animator;

        private void Awake()
        {
            _animator = GetComponentInChildren<Animator>();
        }

        // ── 공개 애니메이션 함수 ─────────────────────────────────────

        /// <summary>걷기 애니메이션</summary>
        public void animation_walk() => PlayAnimation("walk");

        /// <summary>idle(대기) 애니메이션</summary>
        public void animation_idle() => PlayAnimation("idle");

        /// <summary>animation key 기반 실행 ("walk", "idle" 등)</summary>
        public void PlayAnimation(string animationKey)
        {
            if (_animator == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] Animator 없음: {gameObject.name}");
                return;
            }
            _animator.SetTrigger(animationKey);
            Debug.Log($"[CharacterAnimationController] {npcName}: {animationKey}");
        }

        // ── 초기화 ───────────────────────────────────────────────────

        /// <summary>CharacterAssetData로 NPC 정보를 초기화한다.</summary>
        public void Initialize(CharacterAssetData data, string bubble = "")
        {
            if (data == null) return;
            npcName    = data.npc_name;
            bubbleText = bubble;

            // TODO: 유니티 개발자 — texture_url(GLB)로 텍스쳐 적용
            // data.texture_url 이 있으면 glTFast로 GLB 로드 후 Material 추출 → SkinnedMeshRenderer에 적용
            // 예:
            //   if (!string.IsNullOrEmpty(data.texture_url))
            //       StartCoroutine(ApplyTextureFromGlb(data.texture_url));
        }

        // TODO: 유니티 개발자 — glTFast 텍스쳐 적용 구현
        // private IEnumerator ApplyTextureFromGlb(string url)
        // {
        //     var gltf = new GLTFast.GltfImport();
        //     var success = await gltf.Load(url);
        //     if (!success) yield break;
        //     var renderer = GetComponentInChildren<SkinnedMeshRenderer>();
        //     if (renderer != null) renderer.material = gltf.GetMaterial(0);
        // }
    }
}
