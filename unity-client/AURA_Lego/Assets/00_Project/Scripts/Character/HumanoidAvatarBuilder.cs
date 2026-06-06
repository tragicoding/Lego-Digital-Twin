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
    ///  두 가지 네이밍 스킴을 모두 지원한다 (대소문자 무시 매칭):
    ///    1. Mixamo 표준  — "mixamorig:Hips" / "Hips"
    ///    2. Tripo 리깅    — 3ds Max/CAT 스타일 ("Hip", "Waist", "Spine01/02",
    ///       "NeckTwist01", "L_/R_Clavicle·Upperarm·Forearm·Hand·Thigh·Calf·
    ///       Foot·ToeBase")
    ///  ※ Tripo 본 이름은 npc_X_rigged.fbx 의 .meta 및 리깅 GLB(skin joints)에서
    ///    실측 확인됨 (Unity Humanoid 22본 전부 일치). 같은 캐릭터를 FBX(TriLib)
    ///    든 GLB(glTFast)든 로드해도 이 빌더 하나로 Humanoid 아바타 구성 가능.
    /// </summary>
    public static class HumanoidAvatarBuilder
    {
        // (후보 본 이름 배열, Unity HumanBodyBones 이름)
        // 후보는 [Mixamo, Tripo] 순. 한 모델은 둘 중 한 스킴만 쓰므로 충돌 없음.
        // Unity Required: Hips, Spine, Chest, Head, UpperArm×2, LowerArm×2, Hand×2,
        //                 UpperLeg×2, LowerLeg×2, Foot×2
        private static readonly (string[] candidates, string humanName)[] BoneMap =
        {
            // ── 몸통 ──────────────────────────────────────────────────
            (new[] { "Hips", "Hip" },                             "Hips"),
            (new[] { "Spine", "Waist" },                          "Spine"),
            (new[] { "Spine1", "Chest", "Spine01" },              "Chest"),
            (new[] { "Spine2", "UpperChest", "Spine02" },         "UpperChest"),
            (new[] { "Neck", "NeckTwist01" },                     "Neck"),
            (new[] { "Head" },                                    "Head"),

            // ── 왼팔 ──────────────────────────────────────────────────
            (new[] { "LeftShoulder", "L_Clavicle" },              "LeftShoulder"),
            (new[] { "LeftArm", "L_Upperarm" },                   "LeftUpperArm"),
            (new[] { "LeftForeArm", "L_Forearm" },                "LeftLowerArm"),
            (new[] { "LeftHand", "L_Hand" },                      "LeftHand"),

            // ── 오른팔 ────────────────────────────────────────────────
            (new[] { "RightShoulder", "R_Clavicle" },             "RightShoulder"),
            (new[] { "RightArm", "R_Upperarm" },                  "RightUpperArm"),
            (new[] { "RightForeArm", "R_Forearm" },               "RightLowerArm"),
            (new[] { "RightHand", "R_Hand" },                     "RightHand"),

            // ── 왼다리 ────────────────────────────────────────────────
            (new[] { "LeftUpLeg", "L_Thigh" },                    "LeftUpperLeg"),
            (new[] { "LeftLeg", "L_Calf" },                       "LeftLowerLeg"),
            (new[] { "LeftFoot", "L_Foot" },                      "LeftFoot"),
            (new[] { "LeftToeBase", "LeftToe_End", "L_ToeBase" }, "LeftToes"),

            // ── 오른다리 ──────────────────────────────────────────────
            (new[] { "RightUpLeg", "R_Thigh" },                   "RightUpperLeg"),
            (new[] { "RightLeg", "R_Calf" },                      "RightLowerLeg"),
            (new[] { "RightFoot", "R_Foot" },                     "RightFoot"),
            (new[] { "RightToeBase", "RightToe_End", "R_ToeBase" }, "RightToes"),
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
