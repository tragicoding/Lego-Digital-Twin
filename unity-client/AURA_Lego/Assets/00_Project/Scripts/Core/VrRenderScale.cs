using UnityEngine;
using UnityEngine.XR;

namespace LegoTwin.Core
{
    /// <summary>
    /// VR 눈 텍스처 해상도 배율(렌더 스케일)을 모드 확정 시 1회 적용한다 — VR GPU 부하의 1순위 레버.
    /// 렌더할 픽셀 수를 줄여(배율²) 양안 렌더 비용을 직접 낮춘다(고개 돌릴 때 저더·렉 완화).
    ///
    /// 설계 (저결합·확장·서버동일)
    ///   - XRModeManager.Mode 만 1회 읽는다(EventSystemModeBinder 와 동일 패턴). VR일 때만 적용,
    ///     데스크톱은 손대지 않음(평면 렌더라 무관·무회귀). 매니저 없으면 아무것도 안 함.
    ///   - 데이터 소스(Mock/Server) 미참조 → 모드(하드웨어)로만 결정 → 동일 동작.
    ///
    /// 튜닝: 인스펙터 _renderScale 을 바꾸고 다시 Play. GTX1060급/VR은 0.75~0.85 권장(낮출수록 부하↓·선명도↓).
    /// 측정과 병행 권장(PerfHud F9 / OVR Metrics) — GPU 바운드면 효과가 큼.
    /// </summary>
    public class VrRenderScale : MonoBehaviour
    {
        [Tooltip("VR 눈 텍스처 해상도 배율. 1=원본, 낮출수록 GPU 부하↓·선명도↓. GTX1060급은 0.75~0.85 권장.")]
        [SerializeField, Range(0.5f, 1.2f)] private float _renderScale = 0.85f;

        private void Update()
        {
            if (!XRModeManager.Resolved) return;   // 모드 확정 전 대기

            if (XRModeManager.Mode == AppMode.VR)
            {
                XRSettings.eyeTextureResolutionScale = _renderScale;
                Debug.Log($"[VrRenderScale] VR 렌더 스케일 적용: {_renderScale:0.00}");
            }

            enabled = false;   // 1회 적용 후 정지
        }
    }
}
