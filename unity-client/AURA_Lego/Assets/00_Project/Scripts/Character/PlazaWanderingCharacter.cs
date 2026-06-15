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
        private bool    _centerFromSpawn;          // true면 자기 스폰(NavMesh 안착) 위치를 배회 중심으로

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
        private bool      _walking;   // 현재 걷기 애니 상태 — 실제 이동 속도로만 토글

        // ════════════════════════════════════════════════════════════
        // 주입 / 초기화
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// PlazaManager 가 스폰 직후 호출해 배회/인터랙션 파라미터를 주입한다.
        /// centerFromSpawn=true 면 전달된 wanderCenter 대신 '자기 스폰(NavMesh 안착) 위치'를
        /// 배회 중심으로 사용한다 → 캐릭터마다 자기 자리 주변을 돌아 자연히 분산된다.
        /// </summary>
        public void Setup(Vector3 wanderCenter, float wanderRadius, float moveSpeed,
                          float approachRadius, float waitMin, float waitMax,
                          bool centerFromSpawn = false)
        {
            _wanderCenter    = wanderCenter;
            _wanderRadius    = wanderRadius;
            _moveSpeed       = moveSpeed;
            _approachRadius  = approachRadius;
            _waitMin         = waitMin;
            _waitMax         = waitMax;
            _centerFromSpawn = centerFromSpawn;
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

            // 자기 스폰 자리 기준 배회: NavMesh 에 안착한 실제 위치를 배회 중심으로 확정.
            if (_centerFromSpawn) _wanderCenter = transform.position;
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

                // 도착 또는 인터랙션 진입까지 이동.
                // - 걷기/정지 애니는 '실제 이동 속도'로 동기화 → 경로 계산·내비메시 갱신으로
                //   잠깐 멈춰도 제자리 걷기가 아니라 idle 로 보이게 한다.
                // - 오브제 carve 등으로 경로가 무효화되면 즉시 재탐색 → 곧바로 다시 이동.
                while (_state == State.Wandering && AgentReady() &&
                       (_agent.pathPending || _agent.remainingDistance > ArrivalThreshold))
                {
                    if (!_agent.pathPending &&
                        (!_agent.hasPath || _agent.pathStatus == NavMeshPathStatus.PathInvalid))
                        _agent.SetDestination(PickNavPoint());

                    SyncWalkAnim();
                    yield return null;
                }

                if (_state != State.Wandering) continue;

                // 도착 → 잠시 대기
                SetWalking(false);
                if (AgentReady()) _agent.isStopped = true;
                yield return new WaitForSeconds(Random.Range(_waitMin, _waitMax));
            }
        }

        private Vector3 PickNavPoint()
        {
            // 1차: 배회 중심(보통 자기 스폰 자리) 반경 내 NavMesh 지점
            if (TrySampleAround(_wanderCenter, _wanderRadius, out var p)) return p;

            // 2차 폴백: 중심/반경이 NavMesh와 어긋나도 최소한 '현재 위치 주변'에서 움직이게.
            if (TrySampleAround(transform.position, Mathf.Min(_wanderRadius, 5f), out p)) return p;

            return transform.position;
        }

        private static bool TrySampleAround(Vector3 center, float radius, out Vector3 result)
        {
            for (int i = 0; i < 8; i++)
            {
                Vector2 r = Random.insideUnitCircle * radius;
                var candidate = new Vector3(center.x + r.x, center.y, center.z + r.y);
                if (NavMesh.SamplePosition(candidate, out var hit, 4f, NavMesh.AllAreas))
                {
                    result = hit.position;
                    return true;
                }
            }
            result = center;
            return false;
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

            if (_interactRoutine != null) { StopCoroutine(_interactRoutine); _interactRoutine = null; }

            if (_placed != null && _placed.HasSignature)
            {
                // 시그니처 있음 → idle 없이 즉시: 플레이어를 바로 바라보고 시그니처 재생.
                // (현재 자리·바라본 회전을 앵커로 잡아 '멈춰 선 이 자리'에서 동작)
                SnapFacePlayer();
                _placed.RebaseAnchorToCurrent();
                _placed.PlaySignatureMotion();
            }
            else
            {
                // 시그니처 없음 → idle 애니메이션 재생, 부드럽게 플레이어 응시만.
                SetWalking(false);
                _interactRoutine = StartCoroutine(FaceSmooth());
            }
        }

        // 플레이어를 향해 부드럽게 회전(시그니처 없는 idle 캐릭터용). 회전 외 동작 없음.
        private IEnumerator FaceSmooth()
        {
            float t = 0f;
            while (t < FaceDuration)
            {
                FacePlayerStep();
                t += Time.deltaTime;
                yield return null;
            }
            SnapFacePlayer();
            _interactRoutine = null;
        }

        private void EndInteraction()
        {
            if (_interactRoutine != null) { StopCoroutine(_interactRoutine); _interactRoutine = null; }

            _placed?.StopSignatureMotion();
            _walking = false;   // 배회 재개 시 SyncWalkAnim 이 walk 를 다시 켜도록 캐시 리셋

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

        // 실제 Agent 이동 속도로 walk/idle 토글 — 경로 대기/무효로 멈춰 있으면 제자리 걷기 대신 idle.
        private void SyncWalkAnim()
        {
            bool moving = _agent != null && _agent.velocity.sqrMagnitude > 0.04f;  // ≈0.2 m/s
            SetWalking(moving);
        }

        private void SetWalking(bool walk)
        {
            if (walk == _walking) return;
            _walking = walk;
            if (walk) _anim?.animation_walk();
            else      _anim?.animation_idle();
        }

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
