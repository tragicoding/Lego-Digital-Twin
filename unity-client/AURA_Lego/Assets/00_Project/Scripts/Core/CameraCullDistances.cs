using UnityEngine;

namespace LegoTwin.Core
{
    /// <summary>
    /// 카메라 far clip + 레이어별 컬 거리(Camera.layerCullDistances)를 설정해 먼 디테일을 렌더에서 제외한다.
    /// → 드로우콜·삼각형 수를 거리로 직접 깎는다(풀·소품·먼 환경에 효과적).
    ///
    /// 설계 (저결합·무중복·확장·서버동일)
    ///   - 카메라 한 개만 만진다(게임 시스템·SessionManager 미참조). 순수 렌더 설정 → Mock·Server 동일.
    ///   - 컬 거리는 인스펙터 배열로 레이어별 지정(확장: 레이어만 추가). 코드 중복 없음.
    ///   - VR/데스크톱 모드와 무관(지오메트리 비용은 양쪽 공통이므로 항상 적용). 모드 게이트 불필요.
    ///
    /// 사용
    ///   대상 카메라(또는 설정 오브젝트)에 부착. _layers 에 (레이어 이름, 최대 거리) 입력.
    ///   환경/소품/풀을 별도 레이어로 분류해 두면 효과가 큼. 런타임 조정 후 Apply() 재호출 가능.
    ///   ※ 가장 큰 절감은 Occlusion Culling 베이크(에디터) — 이 컴포넌트는 거리 컬링으로 보완.
    /// </summary>
    public class CameraCullDistances : MonoBehaviour
    {
        [System.Serializable]
        public struct LayerCull
        {
            [Tooltip("레이어 이름")] public string layer;
            [Tooltip("이 레이어 오브젝트를 그릴 최대 거리(m). 이보다 멀면 컬링. 0=카메라 far clip 사용")]
            public float distance;
        }

        [Tooltip("대상 카메라(비우면 이 오브젝트의 Camera → Camera.main 순으로 탐색)")]
        [SerializeField] private Camera _camera;

        [Tooltip("카메라 far clip plane(m). 0이면 변경하지 않음. 씬 far가 과도하게 크면 200~500 권장.")]
        [SerializeField] private float _farClip = 0f;

        [Tooltip("구형 컬링 — 거리를 모든 방향에 동일 적용(코너에서 튐 방지). VR 권장.")]
        [SerializeField] private bool _spherical = true;

        [Tooltip("레이어별 컬 거리. 환경·소품·풀 레이어를 가까운 거리로 두면 드로우콜·Tris 절감.")]
        [SerializeField] private LayerCull[] _layers;

        private void Start() => Apply();

        /// <summary>설정을 카메라에 적용한다. 인스펙터 값 변경 후 다시 호출하면 갱신.</summary>
        public void Apply()
        {
            Camera cam = ResolveCamera();
            if (cam == null)
            {
                Debug.LogWarning("[CameraCullDistances] 카메라를 찾지 못했습니다 — 적용 생략.");
                return;
            }

            if (_farClip > 0f) cam.farClipPlane = _farClip;

            // layerCullDistances 는 길이 32 배열을 통째로 받아 수정 후 다시 할당해야 한다(개별 인덱스 직접 설정 불가).
            float[] dists = cam.layerCullDistances;
            if (_layers != null)
            {
                foreach (var lc in _layers)
                {
                    int idx = LayerMask.NameToLayer(lc.layer);
                    if (idx < 0)
                    {
                        Debug.LogWarning($"[CameraCullDistances] 알 수 없는 레이어: \"{lc.layer}\" — 건너뜀");
                        continue;
                    }
                    dists[idx] = Mathf.Max(0f, lc.distance);
                }
            }
            cam.layerCullDistances = dists;
            cam.layerCullSpherical = _spherical;
        }

        private Camera ResolveCamera()
        {
            if (_camera != null) return _camera;
            var own = GetComponent<Camera>();
            return own != null ? own : Camera.main;
        }
    }
}
