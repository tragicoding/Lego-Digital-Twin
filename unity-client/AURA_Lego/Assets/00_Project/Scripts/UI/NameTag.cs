using UnityEngine;

namespace LegoTwin.UI
{
    /// <summary>
    /// 런타임 이름표 — 대상(캐릭터/오브제)의 머리(바운즈 상단) 위에 떠서 항상 카메라를 향한다.
    ///
    /// 설계:
    ///   - 대상의 '자식이 아니라' 독립 오브젝트로 따라다닌다 → 대상 스케일(6·12배) 영향을 안 받음.
    ///   - 대상이 파괴되면(씬 리로드·재스폰) 스스로 정리된다.
    ///   - 표시 전용이라 GraphicRaycaster 가 없다(NameTagSpawner 가 안 붙임) → 입력 비용·NaN 없음.
    ///
    /// 생성은 NameTagSpawner.Attach 가 담당한다(이 컴포넌트를 직접 추가하지 않음).
    /// </summary>
    public class NameTag : MonoBehaviour
    {
        private Transform  _target;
        private Renderer[] _renderers;
        private float      _heightOffset;
        private float      _rightOffset;       // 카메라 기준 좌우 오프셋(+면 화면 오른쪽)
        private float      _forwardOffset;     // 카메라 기준 전후 여유(margin, m). +면 카메라 쪽=앞으로
        private float      _forwardSizeFactor; // 오브제 반(半)깊이 비례 전진 배율(크기 자동 조정). 0이면 고정.
        private bool       _eyeLevel;          // true=플레이어(카메라) 눈높이 기준(오브제 높이 무시)
        private Camera     _cam;

        /// <summary>추적 대상과 오프셋을 주입한다. NameTagSpawner 가 생성 직후 호출.</summary>
        /// <param name="rightOffset">카메라 기준 좌우 이동(m). +면 화면 오른쪽. 0이면 중앙.</param>
        /// <param name="forwardOffset">카메라 기준 전후 여유(m). +면 카메라 쪽(앞). 크기 비례분에 더해지는 고정 margin.</param>
        /// <param name="forwardSizeFactor">오브제 반깊이(카메라 방향) × 이 값만큼 추가로 앞으로. 1=앞면 바로 앞.
        ///   크기에 따라 자동 조정됨. 0이면 forwardOffset 고정값만 사용.</param>
        /// <param name="eyeLevel">true 면 높이를 플레이어(카메라) 눈높이에 맞춤(오브제 높이 무시). heightOffset 은 눈높이 기준 오프셋.</param>
        public void Bind(Transform target, float heightOffset, float rightOffset = 0f,
                         float forwardOffset = 0f, float forwardSizeFactor = 0f, bool eyeLevel = false)
        {
            _target            = target;
            _renderers         = target != null ? target.GetComponentsInChildren<Renderer>() : null;
            _heightOffset      = heightOffset;
            _rightOffset       = rightOffset;
            _forwardOffset     = forwardOffset;
            _forwardSizeFactor = forwardSizeFactor;
            _eyeLevel          = eyeLevel;
            _cam               = Camera.main;
        }

        private void LateUpdate()
        {
            if (_target == null) { Destroy(gameObject); return; }  // 대상 파괴 시 함께 정리

            if (_cam == null) _cam = Camera.main;

            bool hasBounds = TryGetBounds(out Bounds b);

            // 기준 높이: 눈높이(플레이어) 또는 상단(머리 위) + 오프셋. XZ 는 대상 중심.
            float baseY;
            if (_eyeLevel && _cam != null)  baseY = _cam.transform.position.y;  // 플레이어 눈높이
            else if (hasBounds)             baseY = b.max.y;                    // 머리 위(상단)
            else                            baseY = _target.position.y;
            Vector3 pos = new Vector3(_target.position.x, baseY + _heightOffset, _target.position.z);

            // 카메라 기준 좌우/전후 이동(빌보드라 화면상 좌우와 일치)
            if (_cam != null)
            {
                if (_rightOffset != 0f) pos += _cam.transform.right * _rightOffset;

                // 전진 거리 = 오브제 반깊이(카메라 방향) × 배율 + 고정 여유 → 크기에 따라 자동 조정
                float fwd = _forwardOffset;
                if (_forwardSizeFactor != 0f && hasBounds)
                    fwd += HalfDepthAlongCamera(b) * _forwardSizeFactor;
                if (fwd != 0f) pos -= _cam.transform.forward * fwd; // 카메라 쪽=앞
            }
            transform.position = pos;

            // 빌보드 — 항상 카메라와 같은 방향 (World Space Canvas 표준)
            if (_cam != null) transform.forward = _cam.transform.forward;
        }

        // 모든 Renderer 를 합친 월드 바운즈. Renderer 가 없으면 false.
        private bool TryGetBounds(out Bounds bounds)
        {
            bounds = default;
            bool any = false;
            if (_renderers != null)
            {
                foreach (var r in _renderers)
                {
                    if (r == null) continue;
                    if (!any) { bounds = r.bounds; any = true; }
                    else bounds.Encapsulate(r.bounds);
                }
            }
            return any;
        }

        // 카메라 forward 방향으로 본 AABB 반(半)깊이 — 중심에서 앞면까지 거리.
        private float HalfDepthAlongCamera(Bounds b)
        {
            Vector3 e = b.extents;
            Vector3 f = _cam.transform.forward;
            return Mathf.Abs(e.x * f.x) + Mathf.Abs(e.y * f.y) + Mathf.Abs(e.z * f.z);
        }
    }
}
