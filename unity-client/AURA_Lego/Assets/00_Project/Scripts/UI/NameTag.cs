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
        private float      _rightOffset;   // 카메라 기준 좌우 오프셋(+면 화면 오른쪽)
        private Camera     _cam;

        /// <summary>추적 대상과 머리 위 오프셋을 주입한다. NameTagSpawner 가 생성 직후 호출.</summary>
        /// <param name="rightOffset">카메라 기준 좌우 이동(m). +면 화면 오른쪽. 0이면 중앙.</param>
        public void Bind(Transform target, float heightOffset, float rightOffset = 0f)
        {
            _target       = target;
            _renderers    = target != null ? target.GetComponentsInChildren<Renderer>() : null;
            _heightOffset = heightOffset;
            _rightOffset  = rightOffset;
            _cam          = Camera.main;
        }

        private void LateUpdate()
        {
            if (_target == null) { Destroy(gameObject); return; }  // 대상 파괴 시 함께 정리

            if (_cam == null) _cam = Camera.main;

            // 대상 바운즈 상단 + 오프셋 위치로 따라가기 (XZ 는 대상 중심)
            Vector3 pos = new Vector3(_target.position.x, TopY() + _heightOffset, _target.position.z);
            // 카메라 기준 좌우 이동(빌보드라 화면상 좌우와 일치)
            if (_cam != null && _rightOffset != 0f) pos += _cam.transform.right * _rightOffset;
            transform.position = pos;

            // 빌보드 — 항상 카메라와 같은 방향 (World Space Canvas 표준)
            if (_cam != null) transform.forward = _cam.transform.forward;
        }

        // 모든 Renderer 바운즈의 최대 Y(머리 높이). Renderer 가 없으면 대상 위치 Y 폴백.
        private float TopY()
        {
            float top = float.NegativeInfinity;
            if (_renderers != null)
            {
                foreach (var r in _renderers)
                    if (r != null) top = Mathf.Max(top, r.bounds.max.y);
            }
            return float.IsNegativeInfinity(top) ? _target.position.y : top;
        }
    }
}
