using UnityEngine;
using UnityEngine.Events;
using LegoTwin.Core;

namespace LegoTwin.World
{
    /// <summary>
    /// 열기구 탑승 왕복 컨트롤러.
    ///
    /// Enter() : 현재 플레이어 위치를 기억해 두고, 열기구 탑승 위치(ridePoint)로 이동.
    /// Exit()  : 기억해 둔 "처음 위치"로 복귀.
    /// Toggle(): 타고 있으면 Exit, 아니면 Enter (버튼 하나로 토글하고 싶을 때).
    ///
    /// 설계 원칙
    ///   - 단일 소스 재사용 : 실제 이동은 공용 PlayerTeleporter(CC-safe)에 위임 →
    ///                        텔레포트 로직을 다시 짜지 않는다(중복 0).
    ///   - 저결합          : 매니저 싱글톤 미참조. 플레이어 리그와 탑승 위치(Transform)만 안다.
    ///   - 입력/동작 분리  : "어떻게 누르는가"는 모른다. 외부(WorldButton 등)가 Enter/Exit를
    ///                        호출만 하면 된다 → 입력 방식이 바뀌어도 이 클래스는 그대로(확장성).
    ///   - 복귀 위치는 "고정점"이 아니라 "들어온 순간의 위치" → 어디서 타든 그 자리로 정확히 복귀.
    ///   - 이동 추종 방식  : SetParent 대신 LateUpdate 델타 방식.
    ///                        플레이어 리그를 기구 자식으로 붙이면 CharacterController가
    ///                        BasketFloor를 매 프레임 밀어 떨림이 발생하므로 사용하지 않는다.
    ///                        대신 ridePoint 이동 델타를 LateUpdate에서 플레이어에 직접 더한다.
    ///
    /// 유니티 개발자 체크리스트
    ///   [ ] _ridePoint  : 열기구 바구니 안, 플레이어가 설 빈 GameObject 연결
    ///   [ ] _playerRig  : 비우면 "Player" 태그 오브젝트의 루트로 자동 검색
    ///   [ ] Hot Air Balloon.fbx Rigidbody → Is Kinematic ✔
    ///       (비-Kinematic이면 BalloonTour.transform.position을 물리 엔진이 되돌려 기구가 안 움직임)
    ///   주의: 바구니가 공중에 있으면 바닥 Collider가 있어야 한다(FreeCameraController는
    ///         중력을 항상 적용 → 바닥이 없으면 탑승 직후 추락).
    /// </summary>
    public class BalloonRide : MonoBehaviour
    {
        [Header("플레이어")]
        [Tooltip("XR Origin(또는 Camera Rig 루트). 비우면 \"Player\" 태그 오브젝트의 루트로 자동 검색")]
        [SerializeField] private Transform _playerRig;

        [Header("탑승 위치")]
        [Tooltip("열기구 안에서 플레이어가 설 위치(빈 GameObject)")]
        [SerializeField] private Transform _ridePoint;

        [Tooltip("열기구가 움직이는 경우 체크 — LateUpdate마다 ridePoint 이동 델타를 플레이어에 적용한다.\n" +
                 "(SetParent 방식은 CharacterController-Collider 충돌로 떨림이 발생해 사용하지 않음)")]
        [SerializeField] private bool _rideAlongIfMoving = false;

        [Header("이벤트(선택)")]
        [Tooltip("탑승 직전(텔레포트 전) — 여기에 BalloonTour.ResetToStart()를 연결하면 \"열기구 리셋 → 탑승 → 비행\" 흐름이 된다.")]
        public UnityEvent OnBeforeBoard;
        public UnityEvent OnBoarded;   // 탑승 완료 시
        public UnityEvent OnLeft;      // 하차 완료 시

        private bool       _onBoard;
        private Vector3    _returnPos;
        private Quaternion _returnRot;
        private Vector3    _lastRidePointWorld;   // 이전 프레임 ridePoint 월드 위치 (델타 계산용)
        private int        _exitFrame = -1;        // Exit()가 실행된 프레임 번호 (같은 프레임 재진입 방지)

        public bool OnBoard => _onBoard;

        // ── 공개 API (버튼에서 호출) ──────────────────────────────────

        public void Enter()
        {
            if (_onBoard) return;
            // Exit()와 같은 프레임에 BoardingZone이 동시에 트리거되어 즉시 재탑승되는 것을 방지
            if (Time.frameCount == _exitFrame) return;

            var rig = ResolveRig();
            if (rig == null || _ridePoint == null)
            {
                Debug.LogWarning("[BalloonRide] 플레이어 리그 또는 탑승 위치(ridePoint) 미연결 — 탑승 생략");
                return;
            }

            // "처음 위치" 기억 (고정점이 아니라 탑승 직전의 실제 위치)
            _returnPos = rig.position;
            _returnRot = rig.rotation;

            // 탑승 직전 훅 — 열기구를 시작 위치로 리셋(ridePoint도 함께 시작 위치로 이동)한 뒤 탑승.
            OnBeforeBoard?.Invoke();

            PlayerTeleporter.Teleport(rig, _ridePoint.position, _ridePoint.rotation);
            // ridePoint 이동 델타 추종 초기화 (SetParent 대신 LateUpdate에서 적용)
            _lastRidePointWorld = _ridePoint.position;

            _onBoard = true;
            OnBoarded?.Invoke();
        }

        public void Exit()
        {
            if (!_onBoard) return;

            var rig = ResolveRig();
            if (rig == null) return;

            PlayerTeleporter.Teleport(rig, _returnPos, _returnRot);

            _exitFrame = Time.frameCount;   // 이 프레임에 Enter()가 호출되지 않도록 잠금
            _onBoard = false;
            OnLeft?.Invoke();
        }

        public void Toggle()
        {
            if (_onBoard) Exit();
            else          Enter();
        }

        // ── 내부 ─────────────────────────────────────────────────────

        // BalloonTour.Update()가 열기구를 이동시킨 뒤 LateUpdate에서 델타를 플레이어에 적용.
        // Update 시점에 BalloonTour가 이미 위치를 바꿨으므로 LateUpdate에서 정확한 델타를 읽을 수 있다.
        private void LateUpdate()
        {
            if (!_onBoard || !_rideAlongIfMoving || _ridePoint == null) return;

            var rig = ResolveRig();
            if (rig == null) return;

            Vector3 delta = _ridePoint.position - _lastRidePointWorld;
            _lastRidePointWorld = _ridePoint.position;

            if (delta != Vector3.zero)
                rig.position += delta;
        }

        /// <summary>
        /// 텔레포트 등 외부 이동으로 열기구를 이탈했을 때 추종을 즉시 중단한다.
        /// 복귀 이동 없음 — 이미 외부에서 위치가 변경된 상태이므로.
        /// </summary>
        public void ForceAbort()
        {
            _onBoard = false;
        }

        /// <summary>
        /// 씬의 모든 BalloonRide 추종을 강제 중단한다.
        /// PlayerTeleporter 등 Exit()를 거치지 않는 이동 직후 호출한다.
        /// </summary>
        public static void ForceAbortAll()
        {
            foreach (var br in FindObjectsByType<BalloonRide>(FindObjectsSortMode.None))
                br.ForceAbort();
        }

        // _playerRig 미연결 시 "Player" 태그 오브젝트의 루트로 1회 자동 해석 (MapTeleportUI와 동일 패턴)
        private Transform ResolveRig()
        {
            if (_playerRig != null) return _playerRig;

            var tagged = GameObject.FindGameObjectWithTag("Player");
            if (tagged != null) _playerRig = tagged.transform.root;
            return _playerRig;
        }
    }
}
