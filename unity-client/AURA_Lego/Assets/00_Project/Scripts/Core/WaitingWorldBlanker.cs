using UnityEngine;
using LegoTwin.Managers;

namespace LegoTwin.Core
{
    /// <summary>
    /// VR 대기 단계 동안 주변 월드를 가리고 흰 화면만 보이게 한다(영상 집중).
    /// 가이드 시작(START) 전까지 Main Camera 를 SolidColor(흰색) + 지정 레이어만 렌더로 바꾸고,
    /// 시작되면 원래 설정으로 복구한다. 카메라 컬링 방식 — 월드 지오메트리가 통째로 컬링돼 확실히 흰색이 된다.
    ///
    /// 설계(기존 패턴 준수, 저결합):
    ///   - GameFlowManager.GuideStarted(=START) 전이면 블랭크, true 되면 복구(FastSkyDayNight 와 동일 게이트).
    ///   - XRModeManager.Mode == VR 로 확정된 경우에만(데스크톱·Mock/Server 무영향).
    ///   - 씬 리로드(전시 루프) 시 GuideStarted=false 로 리셋 → 다음 관람객도 자동 블랭크.
    ///   - WaitingScreenUI(순수 뷰)·세션은 참조하지 않는다 — 카메라만 이 컴포넌트가 담당.
    ///
    /// 주의: 대기 중 VR 컨트롤러로 키보드·START 를 눌러야 하므로, 보이게 할 레이어(_visibleLayers)에
    ///       UI + 컨트롤러 레이/리티클 레이어를 포함해야 한다(기본 UI). 빠지면 해당 visual 이 안 보인다.
    ///
    /// 씬 구성: 임의의 씬 GameObject(또는 Main Camera)에 부착. 대상 카메라는 비워두면 Camera.main 자동.
    /// </summary>
    public class WaitingWorldBlanker : MonoBehaviour
    {
        [Tooltip("블랭크 중 렌더할 레이어(이 외 월드는 흰색으로 가려짐). UI + 컨트롤러 레이 레이어 포함.")]
        [SerializeField] private LayerMask _visibleLayers = 1 << 5;   // 기본 UI(5)
        [Tooltip("주변을 가릴 배경 색(기본 흰색).")]
        [SerializeField] private Color _blankColor = Color.white;
        [Tooltip("켜면 VR 모드에서만 적용(데스크톱은 평상시 그대로). 끄면 데스크톱에도 적용.")]
        [SerializeField] private bool _onlyInVR = true;
        [Tooltip("대상 카메라(비우면 Camera.main 자동 사용).")]
        [SerializeField] private Camera _camera;

        private Camera _cam;
        private bool _blanked;
        private bool _captured;
        private CameraClearFlags _origFlags;
        private Color _origColor;
        private int  _origMask;

        private void OnDisable() => Restore();   // 비활성화 시 안전 복구

        private void Update()
        {
            var cam = ResolveCamera();
            if (cam == null) return;

            bool shouldBlank = XRModeManager.Resolved
                            && (!_onlyInVR || XRModeManager.Mode == AppMode.VR)
                            && !GameFlowManager.GuideStarted;

            if (shouldBlank && !_blanked) ApplyBlank(cam);
            else if (!shouldBlank && _blanked) Restore();
        }

        private Camera ResolveCamera()
        {
            if (_camera != null) return _camera;
            if (_cam == null) _cam = Camera.main;   // 씬 리로드로 바뀌면 다시 잡음(null 일 때만)
            return _cam;
        }

        private void ApplyBlank(Camera cam)
        {
            if (!_captured)
            {
                _origFlags = cam.clearFlags;
                _origColor = cam.backgroundColor;
                _origMask  = cam.cullingMask;
                _captured  = true;
            }
            cam.clearFlags      = CameraClearFlags.SolidColor;
            cam.backgroundColor = _blankColor;
            cam.cullingMask     = _visibleLayers;
            _blanked = true;
        }

        private void Restore()
        {
            if (!_blanked) return;
            _blanked = false;
            if (!_captured) return;

            var cam = ResolveCamera();
            if (cam == null) return;
            cam.clearFlags      = _origFlags;
            cam.backgroundColor = _origColor;
            cam.cullingMask     = _origMask;
        }
    }
}
