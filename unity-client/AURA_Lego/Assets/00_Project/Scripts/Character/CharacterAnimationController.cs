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
            npcName    = data.npc_name;
            bubbleText = bubble;
        }
    }
}
