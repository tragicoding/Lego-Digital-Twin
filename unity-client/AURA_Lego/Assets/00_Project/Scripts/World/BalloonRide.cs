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
    ///
    /// 유니티 개발자 체크리스트
    ///   [ ] _ridePoint  : 열기구 바구니 안, 플레이어가 설 빈 GameObject 연결
    ///   [ ] _playerRig  : 비우면 "Player" 태그 오브젝트의 루트로 자동 검색
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

        [Tooltip("열기구가 움직이는 경우 체크 — 리그를 ridePoint 자식으로 붙여 함께 이동시킨다.")]
        [SerializeField] private bool _rideAlongIfMoving = false;

        [Header("이벤트(선택)")]
        [Tooltip("탑승 직전(텔레포트 전) — 여기에 BalloonTour.ResetToStart()를 연결하면 \"열기구 리셋 → 탑승 → 비행\" 흐름이 된다.")]
        public UnityEvent OnBeforeBoard;
        public UnityEvent OnBoarded;   // 탑승 완료 시
        public UnityEvent OnLeft;      // 하차 완료 시

        private bool       _onBoard;
        private Vector3    _returnPos;
        private Quaternion _returnRot;
        private Transform  _returnParent;

        public bool OnBoard => _onBoard;

        // ── 공개 API (버튼에서 호출) ──────────────────────────────────

        public void Enter()
        {
            if (_onBoard) return;

            var rig = ResolveRig();
            if (rig == null || _ridePoint == null)
            {
                Debug.LogWarning("[BalloonRide] 플레이어 리그 또는 탑승 위치(ridePoint) 미연결 — 탑승 생략");
                return;
            }

            // "처음 위치" 기억 (고정점이 아니라 탑승 직전의 실제 위치)
            _returnPos    = rig.position;
            _returnRot    = rig.rotation;
            _returnParent = rig.parent;

            // 탑승 직전 훅 — 열기구를 시작 위치로 리셋(ridePoint도 함께 시작 위치로 이동)한 뒤 탑승.
            OnBeforeBoard?.Invoke();

            PlayerTeleporter.Teleport(rig, _ridePoint.position, _ridePoint.rotation);
            if (_rideAlongIfMoving) rig.SetParent(_ridePoint, true);

            _onBoard = true;
            OnBoarded?.Invoke();
        }

        public void Exit()
        {
            if (!_onBoard) return;

            var rig = ResolveRig();
            if (rig == null) return;

            if (_rideAlongIfMoving) rig.SetParent(_returnParent, true);
            PlayerTeleporter.Teleport(rig, _returnPos, _returnRot);

            _onBoard = false;
            OnLeft?.Invoke();
        }

        public void Toggle()
        {
            if (_onBoard) Exit();
            else          Enter();
        }

        // ── 내부 ─────────────────────────────────────────────────────

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
