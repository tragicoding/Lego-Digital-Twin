using System;
using System.Collections.Generic;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// MotionType → AnimationClip 매핑 ScriptableObject.
    ///
    /// 한 MotionType에 여러 클립(기본 clip + variations)을 등록하면,
    /// 같은 동작을 반복 입력할 때마다 직전과 다른 클립이 번갈아 재생된다.
    ///   예) "춤춰줘" 1회차 → Dance, 2회차 → Dance2/Dance3 중 직전 제외 무작위
    ///
    /// 생성 방법:
    ///   Unity 메뉴 → Assets > Create > MINIVERSE > Mixamo Motion Library
    ///
    /// 사용 방법:
    ///   var clip = library.GetClip(MotionType.Dance);                 // 항상 기본 클립
    ///   var clip = library.GetRandomClip(MotionType.Dance, lastClip); // 직전 제외 변형 재생
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] ScriptableObject 인스턴스 생성 후 Assets/00_Project/Animations/ 에 저장
    ///   [ ] Inspector에서 각 MotionType의 Clip(기본)에 AnimationClip 연결
    ///   [ ] 같은 동작의 다른 애니메이션은 Variations 배열에 추가 (선택)
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
            public MotionType motionType;

            [Tooltip("기본 클립 (필수). Assets/00_Project/Animations/Mixamo/ 의 .anim 파일")]
            public AnimationClip clip;

            [Tooltip("같은 동작의 추가 변형 클립들 (선택). " +
                     "여러 개 넣으면 같은 동작을 반복 입력할 때마다 직전과 다른 클립이 재생됩니다.")]
            public AnimationClip[] variations;
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
            new MotionEntry { motionType = MotionType.Cry   },
            new MotionEntry { motionType = MotionType.Walk  },
            new MotionEntry { motionType = MotionType.Handstand },
            new MotionEntry { motionType = MotionType.Hiphop },
            new MotionEntry { motionType = MotionType.Lie },
        };

        // 런타임 룩업 캐시: MotionType → 클립 풀(기본 clip + variations, 중복/null 제거)
        private Dictionary<MotionType, List<AnimationClip>> _cache;

        private void OnEnable() => BuildCache();

        private void BuildCache()
        {
            _cache = new Dictionary<MotionType, List<AnimationClip>>();

            foreach (var entry in _entries)
            {
                if (!_cache.TryGetValue(entry.motionType, out var pool))
                {
                    pool = new List<AnimationClip>();
                    _cache[entry.motionType] = pool;
                }

                AddIfValid(pool, entry.clip);
                if (entry.variations != null)
                {
                    foreach (var v in entry.variations)
                        AddIfValid(pool, v);
                }
            }

            // 클립이 하나도 없는 빈 풀은 제거 (HasClip / 폴백 판정이 정확하도록)
            var empties = new List<MotionType>();
            foreach (var kv in _cache)
                if (kv.Value.Count == 0) empties.Add(kv.Key);
            foreach (var key in empties)
                _cache.Remove(key);
        }

        private static void AddIfValid(List<AnimationClip> pool, AnimationClip clip)
        {
            if (clip != null && !pool.Contains(clip))
                pool.Add(clip);
        }

        /// <summary>
        /// MotionType의 기본(대표) 클립을 반환. 풀의 첫 클립.
        /// 클립이 없으면 Idle 클립을 반환하고, Idle도 없으면 null.
        /// (기존 호출 호환용 — 항상 같은 클립을 반환한다.)
        /// </summary>
        public AnimationClip GetClip(MotionType type)
        {
            var pool = GetClips(type);
            if (pool != null && pool.Count > 0)
                return pool[0];

            if (type != MotionType.Idle)
            {
                var idle = GetClips(MotionType.Idle);
                if (idle != null && idle.Count > 0)
                {
                    Debug.LogWarning($"[MixamoMotionLibrary] '{type}' 클립 없음 → Idle로 대체");
                    return idle[0];
                }
            }

            Debug.LogWarning($"[MixamoMotionLibrary] '{type}' 클립 없음 (Idle도 없음)");
            return null;
        }

        /// <summary>
        /// MotionType의 클립 중 하나를 무작위로 반환하되, exclude(직전 재생 클립)는 제외한다.
        /// 풀에 클립이 1개뿐이면 그 클립을 그대로 반환(중복 불가피).
        /// 풀이 비어 있으면 Idle 풀로 폴백, 그것도 없으면 null.
        ///
        /// 호출 측은 반환값을 다음 호출의 exclude로 넘겨 "직전과 다른 동작"을 보장한다.
        /// </summary>
        public AnimationClip GetRandomClip(MotionType type, AnimationClip exclude = null)
        {
            if (_cache == null) BuildCache();

            _cache.TryGetValue(type, out var pool);

            // 폴백: Idle 풀
            if ((pool == null || pool.Count == 0) && type != MotionType.Idle)
            {
                Debug.LogWarning($"[MixamoMotionLibrary] '{type}' 클립 없음 → Idle로 대체");
                _cache.TryGetValue(MotionType.Idle, out pool);
            }

            if (pool == null || pool.Count == 0)
            {
                Debug.LogWarning($"[MixamoMotionLibrary] '{type}' 클립 없음");
                return null;
            }

            if (pool.Count == 1)
                return pool[0];

            // exclude 인덱스를 건너뛰어 (Count-1)개 중에서 균등 무작위 선택
            int excludeIndex = exclude != null ? pool.IndexOf(exclude) : -1;
            int range        = excludeIndex >= 0 ? pool.Count - 1 : pool.Count;
            int pick         = UnityEngine.Random.Range(0, range);
            if (excludeIndex >= 0 && pick >= excludeIndex)
                pick++;

            return pool[pick];
        }

        /// <summary>
        /// 클립 이름(예: "Dance2")으로 정확한 AnimationClip을 찾아 반환한다.
        /// 시그니처 동작처럼 "관람객이 고른 특정 클립"을 그대로 복원할 때 사용.
        ///
        /// 폴백: 이름이 클립과 일치하지 않지만 MotionType 이름(예: "Dance")이면
        ///       해당 타입의 기본 클립을 반환한다(옛 저장 데이터 호환).
        /// 못 찾으면 null.
        /// </summary>
        public AnimationClip GetClipByName(string clipName)
        {
            if (string.IsNullOrEmpty(clipName)) return null;
            if (_cache == null) BuildCache();

            foreach (var pool in _cache.Values)
                foreach (var clip in pool)
                    if (clip != null && clip.name == clipName)
                        return clip;

            // 폴백: 옛 데이터는 MotionType 이름("Dance")을 저장했음 → 해당 타입 기본 클립
            if (Enum.TryParse(clipName, out MotionType type))
                return GetClip(type);

            Debug.LogWarning($"[MixamoMotionLibrary] '{clipName}' 클립을 찾지 못함");
            return null;
        }

        /// <summary>해당 MotionType의 전체 클립 풀을 반환(없으면 null). 읽기 전용.</summary>
        public IReadOnlyList<AnimationClip> GetClips(MotionType type)
        {
            if (_cache == null) BuildCache();
            return _cache.TryGetValue(type, out var pool) ? pool : null;
        }

        /// <summary>해당 MotionType의 클립이 하나라도 등록되어 있는지 확인.</summary>
        public bool HasClip(MotionType type)
        {
            if (_cache == null) BuildCache();
            return _cache.ContainsKey(type);
        }
    }
}
