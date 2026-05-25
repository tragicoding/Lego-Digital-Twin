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
