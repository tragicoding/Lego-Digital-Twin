using UnityEngine;

namespace LegoTwin.Object
{
    /// <summary>
    /// 낙하한 전시 오브제가 바닥에 안착하면 Rigidbody를 kinematic으로 전환해
    /// 상시 물리 시뮬레이션(Continuous 충돌검사 포함) 비용을 제거한다.
    /// 전환 후 스스로 컴포넌트를 제거한다.
    ///
    /// RuntimePhysicsUtil.SetupSimplePhysics가 자동으로 부착한다.
    /// Mock·Server 어느 경로의 오브제든 동일하게 적용된다.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class PhysicsSettler : MonoBehaviour
    {
        [Tooltip("정지 상태를 이 시간(초)만큼 유지하면 안착으로 판정")]
        public float settleTime = 0.5f;
        [Tooltip("이 속도(m/s) 미만이면 정지로 간주")]
        public float velocityThreshold = 0.05f;
        [Tooltip("안전망: 이 시간(초)이 지나면 무조건 안착 처리")]
        public float maxWaitTime = 10f;

        private Rigidbody _rb;
        private float _stillTimer;
        private float _elapsed;

        private void Awake() => _rb = GetComponent<Rigidbody>();

        private void FixedUpdate()
        {
            if (_rb == null || _rb.isKinematic) { Settle(); return; }

            _elapsed += Time.fixedDeltaTime;

            if (_rb.linearVelocity.sqrMagnitude < velocityThreshold * velocityThreshold)
                _stillTimer += Time.fixedDeltaTime;
            else
                _stillTimer = 0f;

            if (_stillTimer >= settleTime || _elapsed >= maxWaitTime)
                Settle();
        }

        private void Settle()
        {
            if (_rb != null)
            {
                _rb.linearVelocity  = Vector3.zero;
                _rb.angularVelocity = Vector3.zero;
                // Continuous 모드는 kinematic 전환 전에 Discrete로 되돌려야 경고가 없다.
                _rb.collisionDetectionMode = CollisionDetectionMode.Discrete;
                _rb.isKinematic = true;
            }
            Destroy(this);
        }
    }
}
