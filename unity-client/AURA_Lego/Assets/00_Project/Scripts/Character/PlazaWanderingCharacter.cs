using System.Collections;
using UnityEngine;
using UnityEngine.AI;

namespace LegoTwin.Character
{
    /// <summary>
    /// 광장 이전 창작물 캐릭터 — NavMesh 위를 배회(건물·오브제 회피)하다가,
    /// 플레이어가 다가오면 멈춰 서서 플레이어를 바라보고 시그니처 동작을 재생한다.
    /// 플레이어가 멀어지면 동작을 멈추고 다시 배회한다.
    ///
    /// 책임 분리:
    ///   - 이동/회피      : NavMeshAgent (건물은 베이크로, 런타임 오브제는 NavMeshObstacle 로 회피)
    ///   - 시그니처 재생   : PlacedCharacterController (이미 검증된 재생·드리프트 보정 로직 재사용)
    ///   - 걷기/정지 모션  : CharacterAnimationController (walk/idle)
    ///   이 컴포넌트는 위 셋을 "조율"만 하며 Plaza 레이어를 직접 참조하지 않는다(저결합).
    ///
    /// Mock/Server 무관: PlazaManager.SetupCharacterForPlaza(두 모드 공통 합류점) 에서 부착되므로
    ///   데이터 출처와 상관없이 동일하게 동작한다.
    ///
    /// 전제: 광장 바닥에 NavMesh 가 베이크되어 있어야 배회가 동작한다.
    ///   NavMesh 를 못 찾으면 제자리에 머물고, 접근 시 시그니처 인터랙션만 동작한다(그레이스풀 폴백).
    /// </summary>
    [RequireComponent(typeof(PlacedCharacterController))]
    public class PlazaWanderingCharacter : MonoBehaviour
    {
        // 파라미터는 PlazaManager 가 Setup 으로 주입(중앙 관리). 캐릭터별 동일 값.
        private Vector3 _wanderCenter;
        private float   _wanderRadius   = 15f;
        private float   _moveSpeed      = 2.5f;
        private float   _approachRadius = 3.5f;   // 플레이어가 이 거리 안에 들어오면 멈춤·시그니처
        private float   _waitMin        = 1f;
        private float   _waitMax        = 3f;

        private const float ArrivalThreshold   = 0.4f;
        private const float ApproachHysteresis = 1.2f;  // 이탈은 approach+hysteresis (떨림 방지)
        private const float FaceDuration       = 0.35f; // 플레이어를 향해 도는 시간
        private const float NavSampleRange      = 8f;

        private NavMeshAgent                 _agent;       // NavMesh 없으면 null (폴백)
        private PlacedCharacterController     _placed;
        private CharacterAnimationController  _anim;
        private Transform                     _player;

        private enum State { Wandering, Interacting }
        private State     _state = State.Wandering;
        private Coroutine _interactRoutine;

        // ════════════════════════════════════════════════════════════
        // 주입 / 초기화
        // ════════════════════════════════════════════════════════════

        /// <summary>PlazaManager 가 스폰 직후 호출해 배회/인터랙션 파라미터를 주입한다.</summary>
        public void Setup(Vector3 wanderCenter, float wanderRadius, float moveSpeed,
                          float approachRadius, float waitMin, float waitMax)
        {
            _wanderCenter   = wanderCenter;
            _wanderRadius   = wanderRadius;
            _moveSpeed      = moveSpeed;
            _approachRadius = approachRadius;
            _waitMin        = waitMin;
            _waitMax        = waitMax;
        }

        private void Start()
        {
            _placed = GetComponent<PlacedCharacterController>();
            _anim   = GetComponent<CharacterAnimationController>()
                   ?? GetComponentInChildren<CharacterAnimationController>();

            // NavMesh 이동과 충돌하는 루트 모션 제거 — 이동은 Agent 가 담당(제자리 걷기 애니).
            var animator = GetComponentInChildren<Animator>();
            if (animator != null) animator.applyRootMotion = false;

            ResolvePlayer();
            SetupAgent();

            if (_agent != null)
                StartCoroutine(WanderRoutine());
        }

        // NavMesh 가 가까이 있을 때만 Agent 를 추가·배치한다(오프-내비메시 경고 방지).
        private void SetupAgent()
        {
            if (!NavMesh.SamplePosition(transform.position, out var hit, NavSampleRange, NavMesh.AllAreas))
            {
                Debug.LogWarning($"[PlazaWanderingCharacter] NavMesh 를 찾지 못해 배회 비활성화: {name} " +
                                 "(광장 바닥에 NavMesh 를 베이크하세요). 접근 시 시그니처 인터랙션은 동작합니다.");
                return;
            }

            _agent = GetComponent<NavMeshAgent>();
            if (_agent == null) _agent = gameObject.AddComponent<NavMeshAgent>();

            _agent.speed            = _moveSpeed;
            _agent.angularSpeed     = 360f;
            _agent.acceleration     = 12f;
            _agent.stoppingDistance = 0f;
            _agent.radius           = 0.5f;
            _agent.height           = 2f;
            _agent.autoBraking      = true;
            _agent.Warp(hit.position);
        }

        // ════════════════════════════════════════════════════════════
        // 근접 감지 — 상태 전이만 담당
        // ════════════════════════════════════════════════════════════

        private void Update()
        {
            if (_player == null) { ResolvePlayer(); if (_player == null) return; }

            float dist = Vector3.Distance(transform.position, _player.position);

            if (_state == State.Wandering)
            {
                if (dist <= _approachRadius)
                    BeginInteraction();
            }
            else if (dist > _approachRadius + ApproachHysteresis)
            {
                EndInteraction();
            }
        }

        // ════════════════════════════════════════════════════════════
        // 배회 (NavMesh)
        // ════════════════════════════════════════════════════════════

        private IEnumerator WanderRoutine()
        {
            while (true)
            {
                // 인터랙션 중에는 다음 목적지를 잡지 않고 대기
                yield return new WaitUntil(() => _state == State.Wandering);

                if (!AgentReady()) { yield return null; continue; }

                _agent.isStopped = false;
                _agent.SetDestination(PickNavPoint());
                _anim?.animation_walk();

                // 도착 또는 인터랙션 진입까지 이동
                while (_state == State.Wandering && AgentReady() &&
                       (_agent.pathPending || _agent.remainingDistance > ArrivalThreshold))
                    yield return null;

                if (_state != State.Wandering) continue;

                // 도착 → 잠시 대기
                _anim?.animation_idle();
                if (AgentReady()) _agent.isStopped = true;
                yield return new WaitForSeconds(Random.Range(_waitMin, _waitMax));
            }
        }

        private Vector3 PickNavPoint()
        {
            for (int i = 0; i < 8; i++)
            {
                Vector2 r = Random.insideUnitCircle * _wanderRadius;
                var candidate = new Vector3(_wanderCenter.x + r.x, _wanderCenter.y, _wanderCenter.z + r.y);
                if (NavMesh.SamplePosition(candidate, out var hit, 4f, NavMesh.AllAreas))
                    return hit.position;
            }
            return transform.position;
        }

        // ════════════════════════════════════════════════════════════
        // 인터랙션 — 멈춤 → 플레이어 응시 → 시그니처
        // ════════════════════════════════════════════════════════════

        private void BeginInteraction()
        {
            _state = State.Interacting;

            if (AgentReady())
            {
                _agent.isStopped      = true;
                _agent.velocity       = Vector3.zero;
                _agent.updateRotation = false;
                _agent.updatePosition = false;  // 시그니처 동안 transform 은 애니/리베이스가 제어
            }

            _anim?.animation_idle();

            if (_interactRoutine != null) StopCoroutine(_interactRoutine);
            _interactRoutine = StartCoroutine(FaceThenSignature());
        }

        private IEnumerator FaceThenSignature()
        {
            // 플레이어를 향해 부드럽게 회전
            float t = 0f;
            while (t < FaceDuration)
            {
                FacePlayerStep();
                t += Time.deltaTime;
                yield return null;
            }
            SnapFacePlayer();

            // 현재 자리·바라보는 회전을 기준으로 드리프트 앵커를 다시 잡고 시그니처 재생.
            // (스폰 위치가 아니라 '멈춰 선 이 자리'에서 동작하도록)
            _placed?.RebaseAnchorToCurrent();
            _placed?.PlaySignatureMotion();
            _interactRoutine = null;
        }

        private void EndInteraction()
        {
            if (_interactRoutine != null) { StopCoroutine(_interactRoutine); _interactRoutine = null; }

            _placed?.StopSignatureMotion();

            if (_agent != null && _agent.enabled)
            {
                _agent.updatePosition = true;
                _agent.updateRotation = true;
                if (_agent.isOnNavMesh)
                {
                    _agent.Warp(transform.position);  // 인터랙션 동안 분리됐던 위치 재동기화
                    _agent.isStopped = false;
                }
            }

            _state = State.Wandering;  // WanderRoutine 이 다음 목적지를 잡음
        }

        private void FacePlayerStep()
        {
            if (!TryGetYawToPlayer(out var target)) return;
            transform.rotation = Quaternion.Slerp(transform.rotation, target, FaceDuration > 0f ? Time.deltaTime / FaceDuration * 2f : 1f);
        }

        private void SnapFacePlayer()
        {
            if (TryGetYawToPlayer(out var target)) transform.rotation = target;
        }

        private bool TryGetYawToPlayer(out Quaternion yaw)
        {
            yaw = transform.rotation;
            if (_player == null) return false;
            Vector3 dir = _player.position - transform.position;
            dir.y = 0f;
            if (dir.sqrMagnitude < 0.0001f) return false;
            yaw = Quaternion.LookRotation(dir);
            return true;
        }

        // ════════════════════════════════════════════════════════════
        // 유틸
        // ════════════════════════════════════════════════════════════

        private bool AgentReady() => _agent != null && _agent.enabled && _agent.isOnNavMesh;

        private void ResolvePlayer()
        {
            var tagged = GameObject.FindGameObjectWithTag("Player");
            if (tagged != null) { _player = tagged.transform; return; }
            if (Camera.main != null) _player = Camera.main.transform;
        }

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            UnityEditor.Handles.color = new Color(0.2f, 0.6f, 1f, 0.15f);
            UnityEditor.Handles.DrawSolidDisc(_wanderCenter, Vector3.up, _wanderRadius);
            UnityEditor.Handles.color = new Color(1f, 0.8f, 0.2f, 1f);
            UnityEditor.Handles.DrawWireDisc(transform.position, Vector3.up, _approachRadius);
        }
#endif
    }
}
