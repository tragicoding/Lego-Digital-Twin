using UnityEngine;

namespace LegoTwin.UI
{
    /// <summary>
    /// World Space Canvas를 항상 카메라 바로 앞에 띄운다 (HUD 추종형 빌보드).
    /// 힌트 Canvas의 월드 위치가 어디든 관계없이 카메라 앞 _distance 거리에 배치되어
    /// 플레이어 눈앞에서 따라다닌다.
    /// </summary>
    public class HintBillboard : MonoBehaviour
    {
        [Tooltip("카메라 앞 거리 (m)")]
        [SerializeField] private float _distance = 2f;

        [Tooltip("눈높이 기준 수직 오프셋 (m). 음수 = 살짝 아래")]
        [SerializeField] private float _verticalOffset = -0.3f;

        private Camera _cam;

        private void LateUpdate()
        {
            if (_cam == null) _cam = Camera.main;
            if (_cam == null) return;

            Transform ct = _cam.transform;
            transform.position = ct.position
                                 + ct.forward * _distance
                                 + Vector3.up  * _verticalOffset;
            transform.forward  = ct.forward;
        }
    }
}
