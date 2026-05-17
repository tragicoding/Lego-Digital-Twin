using System.Collections.Generic;
using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 서버에서 받은 캐릭터 데이터 기반 애니메이션 컨트롤러.
    /// Unity 개발자는 이 컴포넌트를 캐릭터 루트 GameObject에 추가하고
    /// animation_walk() / animation_Hello() 를 호출한다.
    /// </summary>
    public class CharacterAnimationController : MonoBehaviour
    {
        [Header("Character Info")]
        public string npcName;
        public string bubbleText;

        // 서버에서 받은 애니메이션 메타데이터 (key → AnimationInfo)
        public Dictionary<string, AnimationInfo> availableAnimations
            = new Dictionary<string, AnimationInfo>();

        private Animator _animator;

        private void Awake()
        {
            _animator = GetComponentInChildren<Animator>();
        }

        /// <summary>걷기 애니메이션 실행</summary>
        public void animation_walk()
        {
            PlayAnimation("walk");
        }

        /// <summary>인사_01 애니메이션 실행</summary>
        public void animation_Hello()
        {
            PlayAnimation("hello");
        }

        /// <summary>animation key 기반 실행 ("walk", "hello" 등)</summary>
        public void PlayAnimation(string animationKey)
        {
            if (_animator == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] Animator 없음: {gameObject.name}");
                return;
            }

            if (!availableAnimations.ContainsKey(animationKey))
            {
                Debug.LogWarning($"[CharacterAnimationController] 알 수 없는 animation key: {animationKey}");
                return;
            }

            _animator.SetTrigger(animationKey);
            Debug.Log($"[CharacterAnimationController] {npcName}: {animationKey} 실행");
        }

        /// <summary>서버 CharacterAssetData로 초기화</summary>
        public void Initialize(CharacterAssetData data, string bubble)
        {
            npcName = data.npc_name;
            bubbleText = bubble;
            if (data.animations != null)
                availableAnimations = data.animations;
        }
    }
}
