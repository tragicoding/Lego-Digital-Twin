using System;
using TriLibCore;
using TriLibCore.General;
using TriLibCore.Mappers;
using UnityEngine;
// HumanLimit 은 TriLibCore.General 과 UnityEngine 양쪽에 존재 → TriLib 쪽으로 고정
using HumanLimit = TriLibCore.General.HumanLimit;

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
            foreach (var (bone, names) in TripoBoneMap)
                mapper.AddMapping(bone, new HumanLimit(), names);

            _options = AssetLoader.CreateDefaultLoaderOptions();
            _options.AnimationType       = AnimationType.Humanoid;
            _options.AvatarDefinition    = AvatarDefinitionType.CreateFromThisModel;
            _options.HumanoidAvatarMapper = mapper;
            return _options;
        }

        /// <summary>
        /// FBX URL 을 런타임 다운로드·임포트해 <paramref name="wrapper"/> 아래에 배치한다.
        /// 로드된 모델 루트는 wrapper 의 자식이 되며, Humanoid Animator 가 부착된다.
        /// </summary>
        /// <param name="url">서버 FBX URL (model_url)</param>
        /// <param name="wrapper">위치·스케일이 설정된 컨테이너. 로드 모델의 부모가 된다.</param>
        /// <param name="onLoaded">로드·머티리얼 적용 완료 시 호출</param>
        /// <param name="onError">실패 시 메시지와 함께 호출</param>
        public static void Load(string url, GameObject wrapper,
                                Action onLoaded, Action<string> onError)
        {
            var request = AssetDownloader.CreateWebRequest(url);
            AssetDownloader.LoadModelFromUri(
                request,
                onLoad: null,
                onMaterialsLoad: _ => onLoaded?.Invoke(),
                onProgress: null,
                onError: err => onError?.Invoke(err?.GetInnerException()?.Message ?? "TriLib 로드 실패"),
                wrapperGameObject: wrapper,
                assetLoaderOptions: GetOptions(),
                customContextData: null,
                fileExtension: "fbx");
        }
    }
}
