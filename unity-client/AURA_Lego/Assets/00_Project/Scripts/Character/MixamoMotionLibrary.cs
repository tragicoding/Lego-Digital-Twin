using System;
using System.Collections.Generic;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// MotionType → AnimationClip 매핑 ScriptableObject.
    ///
    /// 생성 방법:
    ///   Unity 메뉴 → Assets > Create > MINIVERSE > Mixamo Motion Library
    ///
    /// 사용 방법:
    ///   var clip = library.GetClip(MotionType.Dance);
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] ScriptableObject 인스턴스 생성 후 Assets/00_Project/Animations/ 에 저장
    ///   [ ] Inspector에서 각 MotionType에 해당하는 AnimationClip 연결
    ///       (AnimationClip 위치: Assets/00_Project/Animations/Mixamo/)
    ///   [ ] PlacedCharacterController.motionLibrary 필드에 이 에셋 연결
    /// </summary>
    [CreateAssetMenu(
        fileName = "MixamoMotionLibrary",
        menuName  = "MINIVERSE/Mixamo Motion Library")]
    public class MixamoMotionLibrary : ScriptableObject
    {
        [Serializable]
        public struct MotionEntry
        {
            public MotionType   motionType;
            [Tooltip("Assets/00_Project/Animations/Mixamo/ 폴더의 .anim 파일")]
            public AnimationClip clip;
        }

        [SerializeField]
        private MotionEntry[] _entries = new MotionEntry[]
        {
            new MotionEntry { motionType = MotionType.Idle  },
            new MotionEntry { motionType = MotionType.Dance },
            new MotionEntry { motionType = MotionType.Run   },
            new MotionEntry { motionType = MotionType.Jump  },
            new MotionEntry { motionType = MotionType.Wave  },
            new MotionEntry { motionType = MotionType.Sit   },
            new MotionEntry { motionType = MotionType.Kick  },
            new MotionEntry { motionType = MotionType.Spin  },
            new MotionEntry { motionType = MotionType.Cheer },
            new MotionEntry { motionType = MotionType.Clap  },
        };

        // 런타임 룩업 캐시
        private Dictionary<MotionType, AnimationClip> _cache;

        private void OnEnable() => BuildCache();

        private void BuildCache()
        {
            _cache = new Dictionary<MotionType, AnimationClip>();
            foreach (var entry in _entries)
            {
                if (entry.clip != null)
                    _cache[entry.motionType] = entry.clip;
            }
        }

        /// <summary>
        /// MotionType에 해당하는 AnimationClip 반환.
        /// 클립이 없으면 Idle 클립을 반환하고, Idle도 없으면 null.
        /// </summary>
        public AnimationClip GetClip(MotionType type)
        {
            if (_cache == null) BuildCache();

            if (_cache.TryGetValue(type, out var clip))
                return clip;

            // 폴백: Idle 반환
            if (type != MotionType.Idle && _cache.TryGetValue(MotionType.Idle, out var idle))
            {
                Debug.LogWarning($"[MixamoMotionLibrary] '{type}' 클립 없음 → Idle로 대체");
                return idle;
            }

            Debug.LogWarning($"[MixamoMotionLibrary] '{type}' 클립 없음 (Idle도 없음)");
            return null;
        }

        /// <summary>해당 MotionType의 클립이 등록되어 있는지 확인.</summary>
        public bool HasClip(MotionType type)
        {
            if (_cache == null) BuildCache();
            return _cache.ContainsKey(type);
        }
    }
}
