using System.Collections;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// NPC 배회 AI — 지정한 중심점 반경 안에서 무작위로 이동한다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] 각 NPC에 Add Component → NPCWander
    ///   [ ] wanderCenter: 배회 중심 Transform 연결 (비워두면 시작 위치 기준)
    ///   [ ] wanderRadius: 배회 반경 조정
    ///   [ ] CharacterAnimationController가 같은 오브젝트에 있으면 walk/idle 자동 연동
    /// </summary>
    public class NPCWander : MonoBehaviour
    {
        [Header("배회 설정")]
        [Tooltip("배회 중심 Transform. 비워두면 시작 위치를 중심으로 사용.")]
        [SerializeField] private Transform wanderCenter;

        [Tooltip("배회 반경 (미터)")]
        [SerializeField] private float wanderRadius = 10f;

        [Tooltip("이동 속도 (m/s)")]
        [SerializeField] private float moveSpeed = 2f;

        [Tooltip("회전 속도")]
        [SerializeField] private float rotationSpeed = 5f;

        [Tooltip("모델 정면 보정 (도). 모델의 얼굴이 로컬 +Z가 아니면 옆/뒤로 걷는다.\n" +
                 "옆으로 걸으면 90 또는 -90, 뒤로 걸으면 180 입력. (정면이 +Z면 0)")]
        [SerializeField] private float modelYawOffset = 0f;

        [Tooltip("목적지 도착 판정 거리")]
        [SerializeField] private float arrivalThreshold = 0.3f;

        [Header("대기 설정")]
        [Tooltip("목적지 도착 후 최소 대기 시간 (초)")]
        [SerializeField] private float waitMin = 1f;

        [Tooltip("목적지 도착 후 최대 대기 시간 (초)")]
        [SerializeField] private float waitMax = 3f;

        private Vector3                   _center;
        private CharacterAnimationController _anim;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Start()
        {
            _center = wanderCenter != null ? wanderCenter.position : transform.position;
            _anim   = GetComponent<CharacterAnimationController>()
                   ?? GetComponentInChildren<CharacterAnimationController>();

            StartCoroutine(WanderRoutine());
        }

        // ════════════════════════════════════════════════════════════
        // 배회 루틴
        // ════════════════════════════════════════════════════════════

        private IEnumerator WanderRoutine()
        {
            while (true)
            {
                // 다음 목적지 선택
                var destination = PickRandomPoint();

                // 이동
                _anim?.animation_walk();
                while (Vector3.Distance(transform.position, destination) > arrivalThreshold)
                {
                    MoveToward(destination);
                    yield return null;
                }

                // 도착 후 대기
                _anim?.animation_idle();
                yield return new WaitForSeconds(Random.Range(waitMin, waitMax));
            }
        }

        private void MoveToward(Vector3 target)
        {
            // 위치 이동
            transform.position = Vector3.MoveTowards(
                transform.position, target, moveSpeed * Time.deltaTime);

            // 방향 회전 (Y축만). modelYawOffset 으로 모델 정면 축(+Z 아님)을 보정한다.
            var dir = new Vector3(target.x - transform.position.x, 0f, target.z - transform.position.z);
            if (dir != Vector3.zero)
                transform.rotation = Quaternion.Slerp(
                    transform.rotation,
                    Quaternion.LookRotation(dir) * Quaternion.Euler(0f, modelYawOffset, 0f),
                    rotationSpeed * Time.deltaTime);
        }

        private Vector3 PickRandomPoint()
        {
            // 중심에서 반경 내 랜덤 XZ 위치, Y는 현재 높이 유지
            var offset = Random.insideUnitCircle * wanderRadius;
            return new Vector3(
                _center.x + offset.x,
                transform.position.y,
                _center.z + offset.y);
        }

        // ════════════════════════════════════════════════════════════
        // 에디터 시각화
        // ════════════════════════════════════════════════════════════

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            var c = wanderCenter != null ? wanderCenter.position : transform.position;
            UnityEditor.Handles.color = new Color(0f, 1f, 0.5f, 0.2f);
            UnityEditor.Handles.DrawSolidDisc(c, Vector3.up, wanderRadius);
            UnityEditor.Handles.color = new Color(0f, 1f, 0.5f, 1f);
            UnityEditor.Handles.DrawWireDisc(c, Vector3.up, wanderRadius);
        }
#endif
    }
}
