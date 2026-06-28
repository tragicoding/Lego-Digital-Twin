using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// 가이드 모드에서 플레이어(XR Origin 루트)가 NPC로부터 일정 거리 이상 멀어지지 않도록 한다.
    ///
    /// maxDistance 이내면 아무것도 하지 않고, 초과할 때만 플레이어를 NPC 방향으로 당긴다.
    /// 방향이나 회전에 의존하지 않으므로 NPC 회전에 영향받지 않는다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] XR Origin(또는 Camera Rig 루트) GameObject에 이 컴포넌트를 부착
    ///   [ ] GameFlowManager Inspector에서 _playerFollowGuide 필드 연결
    ///   [ ] maxDistance / smoothSpeed 씬에서 조정
    /// </summary>
    public class PlayerFollowGuide : MonoBehaviour
    {
        [Header("거리 설정")]
        [Tooltip("NPC로부터 허용되는 최대 거리 (m). 초과 시 플레이어를 당겨옴.")]
        public float maxDistance = 8f;

        [Tooltip("당겨올 때의 보간 속도 (클수록 빠르게 당겨짐)")]
        public float smoothSpeed = 5f;

        private Transform _target;
        private bool      _isFollowing;

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// NPC 거리 유지 시작.
        /// OnGuideFinished 이벤트에 StopFollowing이 자동 연결된다.
        /// </summary>
        public void StartFollowing(GuideNPCController npc)
        {
            _target      = npc.transform;
            _isFollowing = true;
            npc.OnGuideFinished += StopFollowing;
        }

        /// <summary>거리 유지 중단 (OnGuideFinished에 자동 연결됨).</summary>
        public void StopFollowing()
        {
            _isFollowing = false;
            _target      = null;
        }

        // ════════════════════════════════════════════════════════════
        // 업데이트
        // ════════════════════════════════════════════════════════════

        private void LateUpdate()
        {
            if (!_isFollowing || _target == null) return;

            Vector3 toPlayer = transform.position - _target.position;
            float distance   = toPlayer.magnitude;

            // maxDistance 이내면 아무것도 하지 않음
            if (distance <= maxDistance) return;

            // 초과분만큼 NPC 방향으로 당겨서 maxDistance 경계에 위치시킴
            Vector3 desired = _target.position + toPlayer.normalized * maxDistance;
            transform.position = Vector3.Lerp(
                transform.position, desired, smoothSpeed * Time.deltaTime);
        }
    }
}
