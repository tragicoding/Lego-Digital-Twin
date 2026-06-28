using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;   // TrackedDeviceGraphicRaycaster (XRI)

namespace LegoTwin.Core
{
    /// <summary>
    /// 캔버스를 실행 모드(VR/데스크톱)에 맞춰 시작 시 1회 구성한다.
    ///   - 데스크톱 모드 → 아무것도 안 바꾼다(작성된 Screen Space - Overlay 그대로) → 기존 동작 100% 동일.
    ///   - VR 모드      → 컨트롤러 레이로 누를 수 있게 TrackedDeviceGraphicRaycaster 를 보장하고,
    ///                    필요 시 World Space 로 전환해 플레이어 시야 앞에 띄운다.
    ///
    /// 배치(_placement)
    ///   - HeadFollowing : Overlay 캔버스를 World Space 로 바꿔 카메라 앞에 부드럽게 따라붙임(모달/HUD).
    ///   - KeepInPlace   : 이미 World Space 로 월드에 배치된 캔버스(예: 광장 좋아요 패널)는 위치 그대로 두고
    ///                     VR 레이 클릭만 가능하게 raycaster 만 보장.
    ///
    /// 설계
    ///   - XRModeManager(단일 소스)의 정적 Mode/Resolved 만 읽는다(EventSystemModeBinder 와 동일한 1회 패턴, 저결합).
    ///     매니저가 없으면 아무것도 안 한다(무회귀). UI 로직 스크립트는 전혀 참조하지 않는다.
    ///   - 데이터 소스(Mock/Server) 미참조 → 모드(하드웨어)로만 결정 → Mock·Server 동일 동작.
    ///   - 새 UI 는 이 컴포넌트만 캔버스에 붙이면 VR 대응(확장). 데스크톱은 손대지 않아 무회귀.
    ///
    /// 사용: 인터랙티브 캔버스(Canvas 루트)에 부착. Overlay 모달은 HeadFollowing, 월드 고정 UI 는 KeepInPlace.
    /// </summary>
    [RequireComponent(typeof(Canvas))]
    public class ModeAdaptiveCanvas : MonoBehaviour
    {
        private enum Placement { HeadFollowing, KeepInPlace }

        [Header("VR 배치")]
        [Tooltip("HeadFollowing = Overlay→World Space 전환 후 카메라 앞 추적(모달/HUD) / KeepInPlace = 이미 월드 배치된 캔버스는 위치 유지")]
        [SerializeField] private Placement _placement = Placement.HeadFollowing;

        [Header("HeadFollowing 전용 (VR)")]
        [Tooltip("카메라 앞 거리(m)")]
        [SerializeField] private float _distance = 1.5f;
        [Tooltip("World Space 스케일 (Overlay 픽셀 → m 환산). _worldWidthMeters=0 일 때 사용.")]
        [SerializeField] private float _scale = 0.0015f;
        [Tooltip("패널의 목표 물리 폭(m). >0 이면 캡처한 캔버스 폭에 맞춰 스케일을 자동 산출 → 화면 해상도와 무관하게 일정 크기. 0이면 _scale 사용.")]
        [SerializeField] private float _worldWidthMeters = 0f;
        [Tooltip("켜면 CanvasScaler(Scale With Screen Size)의 referenceResolution을 World 크기로 사용 → 작성된 레이아웃이 보존됨(버튼 위치 어긋남 방지). 없으면 _worldSize 폴백.")]
        [SerializeField] private bool _matchCanvasScalerResolution = true;
        [Tooltip("World Space 전환 시 캔버스 RectTransform 크기(px). CanvasScaler 자동 매칭이 불가할 때의 폴백.")]
        [SerializeField] private Vector2 _worldSize = new Vector2(1000f, 700f);
        [Tooltip("머리 추적 부드러움(클수록 빠르게 따라옴). 0이면 첫 배치 후 고정")]
        [SerializeField] private float _followLerp = 8f;
        [Tooltip("기준 카메라(비우면 Camera.main)")]
        [SerializeField] private Camera _camera;

        private Canvas _canvas;
        private bool _applied;
        private bool _follow;

        private void Awake() => _canvas = GetComponent<Canvas>();

        /// <summary>
        /// 런타임에 생성한 World Space 캔버스(예: 광장 좋아요 패널 앵커)에 VR 레이 클릭을 보장한다.
        /// KeepInPlace 로 동작 — VR 모드에서만 TrackedDeviceGraphicRaycaster 를 추가하고 위치는 그대로 둔다.
        /// 데스크톱은 아무것도 하지 않아 기존(마우스) 동작과 100% 동일.
        /// </summary>
        public static ModeAdaptiveCanvas EnsureForWorldCanvas(GameObject canvasGo)
        {
            var c = canvasGo.GetComponent<ModeAdaptiveCanvas>();
            if (c == null) c = canvasGo.AddComponent<ModeAdaptiveCanvas>();
            c._placement = Placement.KeepInPlace;
            return c;
        }

        private void Update()
        {
            if (!_applied)
            {
                if (!XRModeManager.Resolved) return;   // 모드 확정 전엔 대기
                ApplyMode(XRModeManager.Mode);
                _applied = true;
                if (!_follow) enabled = false;         // 추적 안 하면 1회 적용 후 정지
                return;
            }

            if (_follow) FollowHead();
        }

        private void ApplyMode(AppMode mode)
        {
            if (mode != AppMode.VR) return;   // 데스크톱: 손대지 않음(기존 Overlay 그대로)

            if (_placement == Placement.KeepInPlace)
            {
                // 월드 고정 캔버스: 위치/렌더모드는 그대로, VR 레이 클릭만 보장.
                // worldCamera 가 null 인 채로 TrackedDeviceGraphicRaycaster 를 등록하면
                // 이후 worldCamera 가 설정될 때 OnDisable 키 불일치로 KeyNotFoundException 발생.
                // → 등록 전에 worldCamera 를 확정한다.
                if (_canvas.renderMode == RenderMode.WorldSpace && _canvas.worldCamera == null)
                    _canvas.worldCamera = ResolveCamera();
                EnsureTrackedRaycaster();
                return;
            }

            // HeadFollowing: Overlay → World Space 전환 + 카메라 앞 배치.
            // ★ 캔버스 속성(renderMode·worldCamera·sizeDelta)을 모두 바꾼 뒤에 raycaster 를 추가한다.
            //   AddComponent 시점의 OnEnable 등록 키와 OnDisable 해제 키가 일치하려면
            //   등록 전에 캔버스 상태가 최종 확정되어 있어야 한다.
            Camera cam = ResolveCamera();

            // 변환 전에 Overlay 상태의 실제 rect 크기를 캡처 → World sizeDelta 에 그대로 써서
            // 가장자리 앵커·Constant Pixel Size 레이아웃이 데스크톱과 동일하게 보존되도록.
            var rt = (RectTransform)_canvas.transform;
            Vector2 sourceSize = rt.rect.size;

            _canvas.renderMode = RenderMode.WorldSpace;
            _canvas.worldCamera = cam;

            rt.sizeDelta = (sourceSize.x > 1f && sourceSize.y > 1f) ? sourceSize : ResolveWorldSize();

            float scale = (_worldWidthMeters > 0f && rt.sizeDelta.x > 1f)
                ? _worldWidthMeters / rt.sizeDelta.x
                : _scale;
            rt.localScale = Vector3.one * scale;

            EnsureTrackedRaycaster();          // 캔버스 상태 확정 후 raycaster 등록
            SnapInFront(cam);
            _follow = _followLerp > 0f;
        }

        private void FollowHead()
        {
            Camera cam = ResolveCamera();
            if (cam == null) return;

            var ct = cam.transform;
            Vector3 target = ct.position + ct.forward * _distance;
            Quaternion look = Quaternion.LookRotation(target - ct.position);
            float k = Time.unscaledDeltaTime * _followLerp;
            transform.position = Vector3.Lerp(transform.position, target, k);
            transform.rotation = Quaternion.Slerp(transform.rotation, look, k);
        }

        private void SnapInFront(Camera cam)
        {
            if (cam == null) return;
            var ct = cam.transform;
            transform.position = ct.position + ct.forward * _distance;
            transform.rotation = Quaternion.LookRotation(transform.position - ct.position);
        }

        private void EnsureTrackedRaycaster()
        {
            if (GetComponent<TrackedDeviceGraphicRaycaster>() == null)
                gameObject.AddComponent<TrackedDeviceGraphicRaycaster>();
            // 데스크톱용 GraphicRaycaster 는 그대로 둔다(VR에선 XRUIInputModule 이 TrackedDevice 로 처리).
        }

        private Camera ResolveCamera()
        {
            if (_camera != null) return _camera;
            if (_canvas.worldCamera != null) return _canvas.worldCamera;
            return Camera.main;
        }

        /// <summary>
        /// World Space 전환 시 캔버스 rect 크기를 정한다.
        /// CanvasScaler 가 Scale With Screen Size 면 그 referenceResolution(레이아웃이 작성된 좌표계)을
        /// 사용해 자식(버튼 등) 위치가 보존되게 한다. 조건 미충족이면 _worldSize 로 폴백(무회귀).
        /// </summary>
        private Vector2 ResolveWorldSize()
        {
            if (_matchCanvasScalerResolution)
            {
                var scaler = GetComponent<CanvasScaler>();
                if (scaler != null
                    && scaler.uiScaleMode == CanvasScaler.ScaleMode.ScaleWithScreenSize
                    && scaler.referenceResolution.x > 0f
                    && scaler.referenceResolution.y > 0f)
                    return scaler.referenceResolution;
            }
            return _worldSize;
        }
    }
}
