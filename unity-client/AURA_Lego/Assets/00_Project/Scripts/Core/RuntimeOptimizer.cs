using UnityEngine;
using UnityEngine.Rendering;

namespace LegoTwin.Core
{
    /// <summary>
    /// 전시 성능 최적화 — Mock·Server 모드 공통.
    /// 런타임 스폰된 캐릭터/오브제의 렌더러 비용을 낮춘다.
    ///
    /// ── 방침 ────────────────────────────────────────────────────────
    ///  메시(폴리곤)는 원본 그대로 유지한다. decimation 대신 GTX1060급에서
    ///  실제 지배 비용인 "그림자 캐스팅"과 "스키닝 품질"을 낮춰 FPS를 확보한다.
    ///  Mock(저폴리 프리팹)·Server(Tripo 고폴리) 어느 쪽이든 동일하게 호출되며,
    ///  저폴리에는 영향이 미미하고 고폴리에서 큰 이득이 난다.
    ///
    /// ── 조정 ────────────────────────────────────────────────────────
    ///  아래 static 필드를 한 곳에서 바꾸면 모든 스폰 경로에 일괄 반영된다.
    ///  (그림자가 시각적으로 꼭 필요하면 disableShadowCasting=false 로 끄면 됨.)
    /// </summary>
    public static class RuntimeOptimizer
    {
        // 전시 기본값 — 한 곳에서 조정
        public static bool        disableShadowCasting        = true;
        public static bool        disableShadowReceiving      = false;
        public static bool        limitSkinQuality            = true;
        public static SkinQuality skinQuality                 = SkinQuality.Bone2;
        public static bool        disableSkinnedMotionVectors = true;

        /// <summary>
        /// go 아래 모든 Renderer/SkinnedMeshRenderer에 전시용 최적화를 적용한다.
        /// 여러 번 호출해도 안전(idempotent).
        /// </summary>
        public static void Optimize(GameObject go)
        {
            if (go == null) return;

            var renderers = go.GetComponentsInChildren<Renderer>(true);
            foreach (var r in renderers)
            {
                if (r == null) continue;

                if (disableShadowCasting)
                    r.shadowCastingMode = ShadowCastingMode.Off;
                if (disableShadowReceiving)
                    r.receiveShadows = false;

                if (r is SkinnedMeshRenderer smr)
                {
                    if (limitSkinQuality)
                        smr.quality = skinQuality;
                    if (disableSkinnedMotionVectors)
                        smr.skinnedMotionVectors = false;
                }
            }
        }
    }
}
