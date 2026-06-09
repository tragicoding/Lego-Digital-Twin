// TriLib(유료 에셋)은 public 저장소에 커밋하지 않는다(.gitignore).
// 이 파일의 TriLib 의존 코드는 TRILIB define 으로 가드한다.
//   - TriLib 설치 환경: Editor 스크립트(TriLibDefineSetup)가 TRILIB 를 자동 정의 → 실제 로드
//   - 미설치 환경     : TRILIB 미정의 → Load() 가 onError 호출 → 호출부가 Mock 폴백
// 덕분에 TriLib 없이도 프로젝트가 컴파일된다.
#if TRILIB
using TriLibCore;
using TriLibCore.General;
using TriLibCore.Mappers;
// HumanLimit 은 TriLibCore.General 과 UnityEngine 양쪽에 존재 → TriLib 쪽으로 고정
using HumanLimit = TriLibCore.General.HumanLimit;
#endif
using System;
using System.Collections.Generic;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// 서버 캐릭터(Tripo 리깅 FBX)를 TriLib 으로 런타임 로드하는 어댑터.
    ///
    /// ── 의존성 격리 ────────────────────────────────────────────────
    ///  TriLib API(TriLibCore.*) 사용은 이 파일 한 곳으로 한정한다.
    ///  로더를 교체하거나 TriLib 을 제거할 때 이 파일만 수정하면 된다.
    ///  호출부(GeneratedCharacterSpawner)는 TriLib 을 직접 참조하지 않는다.
    ///
    /// ── 동작 ───────────────────────────────────────────────────────
    ///  AnimationType=Humanoid + Tripo 본 매퍼로 FBX 를 Humanoid 아바타까지
    ///  구성해 로드한다. (TriLib 이 T-포즈 보정을 처리 → glTFast 런타임 빌드의
    ///  자세 왜곡 문제 없음.) 텍스처는 별도(texture_url GLB)이므로 여기서는
    ///  스켈레톤·메시만 책임지고, 텍스처는 CharacterAnimationController 가 적용한다.
    /// </summary>
    public static class TripoFbxLoader
    {
#if TRILIB
        // Tripo 리깅 본 이름 → Unity Humanoid 매핑.
        // 출처: npc_X_rigged.fbx 의 .meta humanDescription + 리깅 GLB skin joints 실측(22/22).
        // 각 항목에 Mixamo 표준 이름도 후보로 둬, 다른 리그 규약도 수용한다.
        private static readonly (HumanBodyBones bone, string[] names)[] TripoBoneMap =
        {
            (HumanBodyBones.Hips,          new[] { "Hip", "Hips" }),
            (HumanBodyBones.Spine,         new[] { "Waist", "Spine" }),
            (HumanBodyBones.Chest,         new[] { "Spine01", "Spine1", "Chest" }),
            (HumanBodyBones.UpperChest,    new[] { "Spine02", "Spine2", "UpperChest" }),
            (HumanBodyBones.Neck,          new[] { "NeckTwist01", "Neck" }),
            (HumanBodyBones.Head,          new[] { "Head" }),
            (HumanBodyBones.LeftShoulder,  new[] { "L_Clavicle", "LeftShoulder" }),
            (HumanBodyBones.LeftUpperArm,  new[] { "L_Upperarm", "LeftArm" }),
            (HumanBodyBones.LeftLowerArm,  new[] { "L_Forearm", "LeftForeArm" }),
            (HumanBodyBones.LeftHand,      new[] { "L_Hand", "LeftHand" }),
            (HumanBodyBones.RightShoulder, new[] { "R_Clavicle", "RightShoulder" }),
            (HumanBodyBones.RightUpperArm, new[] { "R_Upperarm", "RightArm" }),
            (HumanBodyBones.RightLowerArm, new[] { "R_Forearm", "RightForeArm" }),
            (HumanBodyBones.RightHand,     new[] { "R_Hand", "RightHand" }),
            (HumanBodyBones.LeftUpperLeg,  new[] { "L_Thigh", "LeftUpLeg" }),
            (HumanBodyBones.LeftLowerLeg,  new[] { "L_Calf", "LeftLeg" }),
            (HumanBodyBones.LeftFoot,      new[] { "L_Foot", "LeftFoot" }),
            (HumanBodyBones.LeftToes,      new[] { "L_ToeBase", "LeftToeBase" }),
            (HumanBodyBones.RightUpperLeg, new[] { "R_Thigh", "RightUpLeg" }),
            (HumanBodyBones.RightLowerLeg, new[] { "R_Calf", "RightLeg" }),
            (HumanBodyBones.RightFoot,     new[] { "R_Foot", "RightFoot" }),
            (HumanBodyBones.RightToes,     new[] { "R_ToeBase", "RightToeBase" }),
        };

        // 매퍼·옵션은 모든 캐릭터가 공유(리그 동일) → 1회만 생성해 캐싱.
        private static AssetLoaderOptions _options;

        private static AssetLoaderOptions GetOptions()
        {
            if (_options != null) return _options;

            var mapper = ScriptableObject.CreateInstance<ByNameHumanoidAvatarMapper>();
            mapper.CaseInsensitive = true;
            var limit = new HumanLimit { useDefaultValues = true };
            foreach (var (bone, names) in TripoBoneMap)
                mapper.AddMapping(bone, limit, names);

            _options = AssetLoader.CreateDefaultLoaderOptions(supressWarning: true);
            _options.AnimationType        = AnimationType.Humanoid;
            _options.AvatarDefinition     = AvatarDefinitionType.CreateFromThisModel;
            _options.HumanoidAvatarMapper = mapper;
            return _options;
        }
#endif

        /// <summary>
        /// FBX URL 을 런타임 다운로드·임포트해 <paramref name="wrapper"/> 아래에 배치한다.
        /// 로드된 모델 루트는 wrapper 의 자식이 되며, Humanoid Animator 가 부착된다.
        /// TriLib 미설치(TRILIB 미정의) 시 onError 를 호출해 호출부가 폴백하도록 한다.
        /// </summary>
        /// <param name="url">서버 FBX URL (model_url)</param>
        /// <param name="wrapper">위치·스케일이 설정된 컨테이너. 로드 모델의 부모가 된다.</param>
        /// <param name="onLoaded">로드·머티리얼 적용 완료 시 호출</param>
        /// <param name="onError">실패·미지원 시 메시지와 함께 호출</param>
        public static void Load(string url, GameObject wrapper,
                                Action onLoaded, Action<string> onError)
        {
#if TRILIB
            var request = AssetDownloader.CreateWebRequest(url);
            AssetDownloader.LoadModelFromUri(
                request,
                onLoad: null,
                onMaterialsLoad: _ =>
                {
                    PostProcessLoadedHierarchy(wrapper);
                    EnsureHumanoidAvatar(wrapper);
                    onLoaded?.Invoke();
                },
                onProgress: null,
                onError: err => onError?.Invoke(err?.GetInnerException()?.Message ?? "TriLib 로드 실패"),
                wrapperGameObject: wrapper,
                assetLoaderOptions: GetOptions(),
                customContextData: null,
                fileExtension: "fbx");
#else
            Debug.LogWarning("[TripoFbxLoader] TRILIB define 없음(TriLib 미설치) — 서버 FBX 로드 불가. " +
                             "Asset Store 에서 TriLib 2 설치 시 자동 활성화됩니다. 지금은 Mock 폴백.");
            onError?.Invoke("TriLib 미설치");
#endif
        }

#if TRILIB
        /// <summary>
        /// TriLib 가 만든 계층에서 스킨 메시 중심으로 정리한다.
        /// Tripo FBX 에서 가끔 보이는 중복/조각 렌더러를 억제해
        /// "정상 캐릭터 1개 + 반쪽 캐릭터 2개" 증상을 완화한다.
        /// </summary>
        private static void PostProcessLoadedHierarchy(GameObject wrapper)
        {
            if (wrapper == null) return;

            DisableStaticRenderersIfSkinnedExists(wrapper);
            DisableFragmentedSkinnedMeshes(wrapper);
            LogLoadedRenderers(wrapper);
        }

        private static void DisableStaticRenderersIfSkinnedExists(GameObject wrapper)
        {
            var skinned = wrapper.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            if (skinned.Length == 0) return;

            foreach (var renderer in wrapper.GetComponentsInChildren<MeshRenderer>(true))
                renderer.enabled = false;
        }

        private static void DisableFragmentedSkinnedMeshes(GameObject wrapper)
        {
            var renderers = wrapper.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            if (renderers.Length < 3) return;

            var primary = FindPrimaryRenderer(renderers);
            if (primary == null) return;

            int primaryVertices = GetVertexCount(primary);
            if (primaryVertices <= 0) return;

            var disabledNames = new List<string>();
            foreach (var candidate in renderers)
            {
                if (candidate == null || candidate == primary) continue;
                if (!ShouldCullAsFragment(primary, candidate, primaryVertices)) continue;

                candidate.enabled = false;
                disabledNames.Add(candidate.name);
            }

            if (disabledNames.Count > 0)
            {
                Debug.LogWarning(
                    $"[TripoFbxLoader] 중복/조각 스킨 메시 비활성화: {string.Join(", ", disabledNames)}");
            }
        }

        private static SkinnedMeshRenderer FindPrimaryRenderer(SkinnedMeshRenderer[] renderers)
        {
            SkinnedMeshRenderer best = null;
            int bestVertices = -1;

            foreach (var renderer in renderers)
            {
                int vertices = GetVertexCount(renderer);
                if (vertices <= bestVertices) continue;
                best = renderer;
                bestVertices = vertices;
            }

            return best;
        }

        private static int GetVertexCount(SkinnedMeshRenderer renderer)
        {
            return renderer != null && renderer.sharedMesh != null
                ? renderer.sharedMesh.vertexCount
                : 0;
        }

        private static bool ShouldCullAsFragment(
            SkinnedMeshRenderer primary,
            SkinnedMeshRenderer candidate,
            int primaryVertices)
        {
            int candidateVertices = GetVertexCount(candidate);
            if (candidateVertices <= 0) return false;

            bool similarSkeleton =
                (primary.rootBone == null && candidate.rootBone == null) ||
                primary.rootBone == candidate.rootBone;

            if (!similarSkeleton) return false;
            if (candidateVertices > primaryVertices * 0.65f) return false;

            float overlap = BoundsOverlapRatio(primary.bounds, candidate.bounds);
            float sizeRatio = SafeDivide(candidate.bounds.size.magnitude, primary.bounds.size.magnitude);
            bool centeredNearPrimary =
                Vector3.Distance(primary.bounds.center, candidate.bounds.center) <=
                primary.bounds.extents.magnitude * 0.35f;

            return overlap >= 0.55f && sizeRatio >= 0.35f && centeredNearPrimary;
        }

        private static float BoundsOverlapRatio(Bounds a, Bounds b)
        {
            Vector3 min = Vector3.Max(a.min, b.min);
            Vector3 max = Vector3.Min(a.max, b.max);
            Vector3 size = max - min;
            if (size.x <= 0f || size.y <= 0f || size.z <= 0f) return 0f;

            float intersection = size.x * size.y * size.z;
            float bVolume = b.size.x * b.size.y * b.size.z;
            return SafeDivide(intersection, bVolume);
        }

        private static float SafeDivide(float numerator, float denominator)
        {
            return denominator > 0f ? numerator / denominator : 0f;
        }

        private static void EnsureHumanoidAvatar(GameObject wrapper)
        {
            var animator = FindOrCreateAnimator(wrapper);
            if (animator == null)
            {
                Debug.LogWarning($"[TripoFbxLoader] Animator를 찾거나 만들지 못함: {wrapper.name}");
                return;
            }

            var avatar = animator.avatar;
            bool hadValidTriLibAvatar = avatar != null && avatar.isValid && avatar.isHuman;
            if (HumanoidAvatarBuilder.Build(animator.gameObject))
            {
                Debug.LogWarning(hadValidTriLibAvatar
                    ? $"[TripoFbxLoader] TriLib Avatar를 HumanoidAvatarBuilder로 교체: {animator.gameObject.name}"
                    : $"[TripoFbxLoader] TriLib Avatar 비정상 → HumanoidAvatarBuilder fallback 적용: {animator.gameObject.name}");
                return;
            }

            Debug.LogWarning(
                $"[TripoFbxLoader] HumanoidAvatarBuilder 적용 실패, TriLib Avatar 유지: {animator.gameObject.name}");
        }

        private static Animator FindOrCreateAnimator(GameObject wrapper)
        {
            if (wrapper == null) return null;

            var animator = wrapper.GetComponentInChildren<Animator>(true);
            if (animator != null) return animator;

            var renderers = wrapper.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            foreach (var renderer in renderers)
            {
                if (renderer == null) continue;

                var host = FindAnimatorHost(wrapper.transform, renderer.rootBone, renderer.transform);
                if (host == null) continue;

                animator = host.GetComponent<Animator>();
                if (animator == null)
                    animator = host.gameObject.AddComponent<Animator>();

                Debug.LogWarning(
                    $"[TripoFbxLoader] Animator 없음 → 스켈레톤 루트에 Animator 추가: {host.name}");
                return animator;
            }

            return null;
        }

        private static Transform FindAnimatorHost(
            Transform wrapperRoot,
            Transform rootBone,
            Transform rendererTransform)
        {
            var candidate = rootBone != null ? rootBone : rendererTransform;
            if (candidate == null) return null;

            while (candidate.parent != null && candidate.parent != wrapperRoot)
                candidate = candidate.parent;

            return candidate;
        }

        private static void LogLoadedRenderers(GameObject wrapper)
        {
            var renderers = wrapper.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            if (renderers.Length == 0)
            {
                Debug.LogWarning($"[TripoFbxLoader] SkinnedMeshRenderer 없음: {wrapper.name}");
                return;
            }

            var parts = new List<string>(renderers.Length);
            foreach (var renderer in renderers)
            {
                parts.Add(
                    $"{renderer.name}(verts={GetVertexCount(renderer)}, enabled={renderer.enabled}, rootBone={renderer.rootBone?.name ?? "null"})");
            }

            Debug.Log($"[TripoFbxLoader] 로드된 스킨 메시 {renderers.Length}개 — {string.Join(" | ", parts)}");
        }
#endif
    }
}
