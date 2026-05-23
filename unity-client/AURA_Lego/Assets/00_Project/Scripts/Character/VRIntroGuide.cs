using System.Collections;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Locomotion;

namespace LegoTwin.Character
{
    /// <summary>
    /// VR 인트로 가이드 — XR Origin을 시작 위치에서 목표 위치까지 자동 이동시킨다.
    ///
    /// 인트로 전 과정(대기 + 이동)에서 LocomotionProvider를 비활성화해
    /// 컨트롤러 이동/회전을 차단한다. HMD 시선 추적은 항상 유지된다.
    ///
    /// Inspector 설정:
    ///   1. XR Origin 필드에 XR Origin GameObject를 연결한다.
    ///   2. Start Point / Target Point Transform을 씬에서 지정한다.
    ///   3. Wait Duration / Move Duration(초)을 조정한다.
    ///   4. Locomotion Providers 배열이 비어있으면 XR Origin 하위에서 자동 탐색한다.
    /// </summary>
    public class VRIntroGuide : MonoBehaviour
    {
        [Header("XR Origin")]
        [Tooltip("이동 대상 XR Origin. 비워두면 이 GameObject를 사용.")]
        public Transform xrOrigin;

        [Header("위치 설정")]
        [Tooltip("인트로 시작 위치. 비워두면 씬 로드 시 현재 위치를 시작점으로 사용.")]
        public Transform startPoint;

        [Tooltip("인트로 종료 후 도달할 목표 위치. (필수)")]
        public Transform targetPoint;

        [Header("시간 설정 (초)")]
        [Tooltip("이동 전 시작 위치에서 대기하는 시간")]
        public float waitDuration = 5f;

        [Tooltip("목표 위치까지 이동하는 데 걸리는 시간")]
        public float moveDuration = 15f;

        [Header("이동 잠금 (XR Locomotion Providers)")]
        [Tooltip("인트로 동안 비활성화할 LocomotionProvider 목록.\n" +
                 "비워두면 xrOrigin 하위 컴포넌트를 자동 탐색한다.\n" +
                 "HMD 시선 추적은 LocomotionProvider와 무관하므로 항상 유지된다.")]
        public LocomotionProvider[] locomotionProviders;

        private Transform _moveTarget;
        private LocomotionProvider[] _resolvedProviders;
        private bool[] _savedStates;

        private void Start()
        {
            _moveTarget = xrOrigin != null ? xrOrigin : transform;

            if (targetPoint == null)
            {
                Debug.LogError("[VRIntroGuide] targetPoint가 설정되지 않았습니다.");
                return;
            }

            StartCoroutine(IntroRoutine());
        }

        private IEnumerator IntroRoutine()
        {
            // 1. 시작 위치 설정
            if (startPoint != null)
                _moveTarget.SetPositionAndRotation(startPoint.position, startPoint.rotation);

            // 2. 컨트롤러 이동/회전 잠금 (HMD 시선 추적은 유지됨)
            SetLocomotionEnabled(false);

            // 3. 대기
            Debug.Log($"[VRIntroGuide] 대기 시작 ({waitDuration}초)");
            yield return new WaitForSeconds(waitDuration);

            // 4. 이동
            Vector3 fromPos = _moveTarget.position;
            Quaternion fromRot = _moveTarget.rotation;
            float elapsed = 0f;

            Debug.Log($"[VRIntroGuide] 이동 시작 → {targetPoint.name} ({moveDuration}초)");

            while (elapsed < moveDuration)
            {
                float t = Mathf.SmoothStep(0f, 1f, elapsed / moveDuration);
                _moveTarget.SetPositionAndRotation(
                    Vector3.Lerp(fromPos, targetPoint.position, t),
                    Quaternion.Slerp(fromRot, targetPoint.rotation, t)
                );
                elapsed += Time.deltaTime;
                yield return null;
            }

            // 5. 목표 위치 정확히 안착
            _moveTarget.SetPositionAndRotation(targetPoint.position, targetPoint.rotation);

            // 6. 이동 잠금 해제 (원래 활성화 상태로 복원)
            SetLocomotionEnabled(true);

            Debug.Log("[VRIntroGuide] 인트로 가이드 완료.");
        }

        private void SetLocomotionEnabled(bool enable)
        {
            if (enable)
            {
                // 인트로 전 원래 상태 복원
                if (_resolvedProviders == null) return;

                for (int i = 0; i < _resolvedProviders.Length; i++)
                {
                    if (_resolvedProviders[i] != null)
                        _resolvedProviders[i].enabled = _savedStates[i];
                }

                Debug.Log($"[VRIntroGuide] LocomotionProvider 복원 ({_resolvedProviders.Length}개)");
            }
            else
            {
                // 현재 활성화 상태를 저장한 뒤 비활성화
                _resolvedProviders = (locomotionProviders != null && locomotionProviders.Length > 0)
                    ? locomotionProviders
                    : _moveTarget.GetComponentsInChildren<LocomotionProvider>(includeInactive: true);

                _savedStates = new bool[_resolvedProviders.Length];

                for (int i = 0; i < _resolvedProviders.Length; i++)
                {
                    if (_resolvedProviders[i] != null)
                    {
                        _savedStates[i] = _resolvedProviders[i].enabled;
                        _resolvedProviders[i].enabled = false;
                    }
                }

                Debug.Log($"[VRIntroGuide] LocomotionProvider 비활성화 ({_resolvedProviders.Length}개)");
            }
        }
    }
}
