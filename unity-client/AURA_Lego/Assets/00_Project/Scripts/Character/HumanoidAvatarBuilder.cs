using System.Collections.Generic;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// 서버에서 런타임 로드한 캐릭터 GameObject 에 Humanoid Avatar 를 자동 구성합니다.
    ///
    /// ── 사용 시점 ──────────────────────────────────────────────────
    ///  - Mock Prefab: Editor 임포트 시 CharacterRigImporter 가 처리 (이 클래스 불필요)
    ///  - Server FBX : TriLib/glTFast 로드 완료 콜백 안에서 Build(go) 호출
    ///
    /// ── 지원 본 이름 ───────────────────────────────────────────────
    ///  Mixamo 표준 두 가지 형식 모두 지원:
    ///    1. 접두어 있음  — "mixamorig:Hips"
    ///    2. 접두어 없음  — "Hips"
    ///  대소문자 무시(case-insensitive) 매칭.
    /// </summary>
    public static class HumanoidAvatarBuilder
    {
        // (후보 본 이름 배열, Unity HumanBodyBones 이름)
        // Unity Required: Hips, Spine, Chest, Head, UpperArm×2, LowerArm×2, Hand×2,
        //                 UpperLeg×2, LowerLeg×2, Foot×2
        private static readonly (string[] candidates, string humanName)[] BoneMap =
        {
            // ── 몸통 ──────────────────────────────────────────────────
            (new[] { "Hips" },                                    "Hips"),
            (new[] { "Spine" },                                   "Spine"),
            (new[] { "Spine1", "Chest" },                         "Chest"),
            (new[] { "Spine2", "UpperChest" },                    "UpperChest"),
            (new[] { "Neck" },                                    "Neck"),
            (new[] { "Head" },                                    "Head"),

            // ── 왼팔 ──────────────────────────────────────────────────
            (new[] { "LeftShoulder" },                            "LeftShoulder"),
            (new[] { "LeftArm" },                                 "LeftUpperArm"),
            (new[] { "LeftForeArm" },                             "LeftLowerArm"),
            (new[] { "LeftHand" },                                "LeftHand"),

            // ── 오른팔 ────────────────────────────────────────────────
            (new[] { "RightShoulder" },                           "RightShoulder"),
            (new[] { "RightArm" },                                "RightUpperArm"),
            (new[] { "RightForeArm" },                            "RightLowerArm"),
            (new[] { "RightHand" },                               "RightHand"),

            // ── 왼다리 ────────────────────────────────────────────────
            (new[] { "LeftUpLeg" },                               "LeftUpperLeg"),
            (new[] { "LeftLeg" },                                 "LeftLowerLeg"),
            (new[] { "LeftFoot" },                                "LeftFoot"),
            (new[] { "LeftToeBase", "LeftToe_End" },              "LeftToes"),

            // ── 오른다리 ──────────────────────────────────────────────
            (new[] { "RightUpLeg" },                              "RightUpperLeg"),
            (new[] { "RightLeg" },                                "RightLowerLeg"),
            (new[] { "RightFoot" },                               "RightFoot"),
            (new[] { "RightToeBase", "RightToe_End" },            "RightToes"),
        };

        // Mixamo 접두어 변형 목록
        private static readonly string[] Prefixes = { "mixamorig:", "mixamorig1:", "" };

        /// <summary>
        /// <paramref name="root"/> 의 본 계층을 Mixamo 이름으로 매핑하여
        /// Humanoid Avatar 를 빌드하고 Animator 에 적용합니다.
        /// </summary>
        /// <returns>빌드 성공 여부</returns>
        public static bool Build(GameObject root)
        {
            var animator = root.GetComponent<Animator>();
            if (animator == null)
                animator = root.AddComponent<Animator>();

            // 전체 자식 Transform 을 이름→Transform 딕셔너리로 캐싱
            var transforms = root.GetComponentsInChildren<Transform>(true);
            var boneDict   = new Dictionary<string, Transform>(
                System.StringComparer.OrdinalIgnoreCase);

            foreach (var t in transforms)
                boneDict.TryAdd(t.name, t);

            // ── HumanBone 매핑 ──────────────────────────────────────
            var humanBones = new List<HumanBone>();

            foreach (var (candidates, humanName) in BoneMap)
            {
                Transform found = FindBone(boneDict, candidates);
                if (found == null) continue;

                humanBones.Add(new HumanBone
                {
                    boneName  = found.name,
                    humanName = humanName,
                    limit     = new HumanLimit { useDefaultValues = true }
                });
            }

            if (humanBones.Count == 0)
            {
                Debug.LogWarning(
                    $"[HumanoidAvatarBuilder] 본 매핑 실패 — Mixamo 표준 이름이 아닐 수 있습니다: {root.name}");
                return false;
            }

            // ── SkeletonBone (전체 계층 T-포즈 정보) ─────────────────
            var skeletonBones = new SkeletonBone[transforms.Length];
            for (int i = 0; i < transforms.Length; i++)
            {
                var t = transforms[i];
                skeletonBones[i] = new SkeletonBone
                {
                    name     = t.name,
                    position = t.localPosition,
                    rotation = t.localRotation,
                    scale    = t.localScale
                };
            }

            // ── HumanDescription 조립 ────────────────────────────────
            var desc = new HumanDescription
            {
                human             = humanBones.ToArray(),
                skeleton          = skeletonBones,
                upperArmTwist     = 0.5f,
                lowerArmTwist     = 0.5f,
                upperLegTwist     = 0.5f,
                lowerLegTwist     = 0.5f,
                armStretch        = 0.05f,
                legStretch        = 0.05f,
                feetSpacing       = 0f,
                hasTranslationDoF = false
            };

            var avatar = AvatarBuilder.BuildHumanAvatar(root, desc);
            if (!avatar.isValid)
            {
                Debug.LogWarning(
                    $"[HumanoidAvatarBuilder] Avatar 빌드 실패 (isValid=false): {root.name}\n" +
                    $"  매핑된 본: {humanBones.Count}개");
                return false;
            }

            avatar.name    = $"{root.name}_HumanoidAvatar";
            animator.avatar = avatar;

            Debug.Log(
                $"[HumanoidAvatarBuilder] Humanoid Avatar 빌드 성공: {root.name} " +
                $"({humanBones.Count}개 본 매핑)");
            return true;
        }

        // ════════════════════════════════════════════════════════════
        // 유틸
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 후보 본 이름 배열 × Mixamo 접두어 조합으로 Transform 탐색.
        /// </summary>
        private static Transform FindBone(
            Dictionary<string, Transform> dict,
            string[] candidates)
        {
            foreach (var prefix in Prefixes)
            foreach (var name   in candidates)
            {
                if (dict.TryGetValue(prefix + name, out var t)) return t;
            }
            return null;
        }
    }
}
