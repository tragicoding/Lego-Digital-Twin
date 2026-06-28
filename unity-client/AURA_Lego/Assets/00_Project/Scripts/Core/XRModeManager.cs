using System;
using System.Collections;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Management;

namespace LegoTwin.Core
{
    /// <summary>
    /// VR(테더드 HMD) ↔ 데스크톱(키보드+마우스) 실행 모드를 시작 시 1회 판정해 고정하는 단일 소스.
    ///
    /// 설계 원칙
    ///   - 단일 소스      : 모드는 여기서만 결정. 다른 시스템은 정적 Mode/Resolved 를 읽거나
    ///                      ModeResolved 이벤트를 구독해 "스스로" 구성한다.
    ///   - 저결합         : 이 매니저는 FreeCameraController·EventSystem·XR 리그를 직접 참조하지 않는다.
    ///                      (반응 책임은 각 컴포넌트에 위임 → 새 시스템은 이벤트만 구독하면 확장됨)
    ///   - 데이터 소스 무관: SessionManager·dataSourceMode 를 일절 참조하지 않는다.
    ///                      모드는 하드웨어(HMD 착용)로만 정해지므로 Mock·Server 동일 동작.
    ///   - 1회 판정 고정   : "시작/대기화면에서 한 번" 판정 후 세션 동안 고정(사용자 결정).
    ///
    /// 판정 규칙(Auto)
    ///   - 착용(userPresence=true) 감지 → 즉시 VR 확정.
    ///   - XR 초기화 유예(_minSettle) 후에도 활성 HMD 없음 → 즉시 데스크톱 확정(미연결).
    ///   - HMD는 연결됐으나 미착용 → _settleTimeout 까지 착용 대기, 끝나면 데스크톱.
    ///   - 데스크톱 확정 시 XR 서브시스템을 중지해 평면 렌더링/오버헤드 제거(_stopXrInDesktop).
    ///
    /// 전제(에디터 설정)
    ///   PC(Standalone) XR Plug-in Management 에 OpenXR(또는 Oculus) 로더 + Initialize XR on Startup.
    ///   미설정 시 XR 미초기화 → 항상 데스크톱으로 확정(기존 키마 동작 그대로, 무회귀).
    /// </summary>
    public class XRModeManager : MonoBehaviour
    {
        public static XRModeManager Instance { get; private set; }

        /// <summary>확정된 실행 모드. 확정 전 기본값은 DesktopKBM.</summary>
        public static AppMode Mode { get; private set; } = AppMode.DesktopKBM;

        /// <summary>모드가 1회 판정으로 확정됐는지.</summary>
        public static bool Resolved { get; private set; }

        /// <summary>확정 순간 1회 발행(이미 확정된 뒤 구독자는 Resolved 를 직접 확인).</summary>
        public static event Action<AppMode> ModeResolved;

        private enum ModeOverride { Auto, ForceVR, ForceDesktop }

        [Header("판정")]
        [Tooltip("Auto = 착용 자동 감지 / Force* = 헤드셋 없이 에디터에서 모드 강제(테스트용)")]
        [SerializeField] private ModeOverride _override = ModeOverride.Auto;

        [Tooltip("XR 런타임 초기화를 기다리는 최소 유예(초). 이 시간 전에는 '미연결' 판정을 보류한다.")]
        [SerializeField] private float _minSettle = 0.5f;

        [Tooltip("HMD가 연결됐으나 미착용일 때 착용을 기다리는 최대 시간(초). 초과 시 데스크톱으로 확정.")]
        [SerializeField] private float _settleTimeout = 2f;

        [Header("데스크톱 모드")]
        [Tooltip("데스크톱으로 확정될 때 XR 서브시스템을 중지(평면 렌더링·VR 오버헤드 제거).")]
        [SerializeField] private bool _stopXrInDesktop = true;

        [Header("대기 중 XR 재획득")]
        [Tooltip("대기영상(게이트) 동안 XR이 안 떠 있으면(기기 미연결·Quest Link 끊김) 주기적으로 XR 로더 초기화를 재시도한다.\n" +
                 "도중에 재연결+착용하면 VR로 전환된다. (전제: XR Plug-in Management(PC)에 Initialize XR on Startup)")]
        [SerializeField] private bool _reacquireXrWhileGated = true;

        [Tooltip("XR 로더 재초기화 재시도 간격(초). 너무 짧으면 미연결 시 에러 로그가 잦아진다.")]
        [SerializeField] private float _xrRetryInterval = 2f;

        // 대기(attract) 게이트. true면 자동 데스크톱 확정을 보류하고 착용 시에만 즉시 VR로 확정한다.
        // 데스크톱 확정은 외부 게이트의 ResolveNow() 호출로만 이뤄진다(예: 대기영상 종료). 확정 후엔 무의미.
        private bool _gated;

        private void Awake()
        {
            // 씬 리로드(A안 전시 루프)에도 한 번 정해진 모드를 유지하기 위해 DontDestroyOnLoad 싱글톤.
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
            transform.SetParent(null);
            DontDestroyOnLoad(gameObject);

            if (Resolved) return;                 // 리로드 재진입: 이미 확정됨(재판정 안 함)
            StartCoroutine(ResolveRoutine());
        }

        /// <summary>
        /// 외부 게이트가 "지금부터 모드를 확정하지 말고 착용 감지를 계속 열어두라"고 알릴 때 호출
        /// (예: 대기영상 시작). 이 상태에선 미착용이어도 자동으로 데스크톱 확정하지 않고,
        /// 착용되는 즉시 VR로 확정한다. 데스크톱 확정은 게이트가 ResolveNow()를 호출하는 시점에 이뤄진다.
        /// 이미 확정됐으면 무시(리로드·무회귀 안전). 호출자(대기화면)와의 결합은 이 한 줄뿐이다.
        /// </summary>
        public void BeginGatedResolve()
        {
            if (Resolved) return;
            _gated = true;
        }

        /// <summary>
        /// 외부 게이트(예: 대기영상 종료 = 이름 입력으로 넘어가는 순간)에서 모드를 즉시 확정한다.
        /// 그때까지 착용 중이면 VR, 아니면 데스크톱으로 확정. 이미 확정됐으면 무시.
        /// </summary>
        public void ResolveNow()
        {
            if (Resolved) return;
            StopAllCoroutines();
            LockMode(DecideNow());
        }

        private IEnumerator ResolveRoutine()
        {
            if (_override != ModeOverride.Auto)
            {
                LockMode(_override == ModeOverride.ForceVR ? AppMode.VR : AppMode.DesktopKBM);
                yield break;
            }

            float t = 0f;
            float nextXrTry = 0f;
            while (true)
            {
                // 착용은 게이트와 무관하게 언제든 즉시 VR로 확정한다(대기영상 도중 착용 포함).
                if (IsHmdWorn()) { LockMode(AppMode.VR); yield break; }

                if (_gated)
                {
                    // 게이트(대기영상) 동안: 자동 데스크톱 확정을 보류하고 착용을 계속 기다린다.
                    // 데스크톱 확정은 외부 게이트의 ResolveNow() 가 담당한다(예: 대기영상 종료).
                    // XR이 안 떠 있으면(기기 미연결·Quest Link 끊김) 주기적으로 로더 재초기화를 시도해,
                    // 대기영상 도중 재연결되면 위 IsHmdWorn 에서 VR로 확정되게 한다.
                    // (_minSettle 이후에만 시도 → 부팅 시 Initialize-on-Startup 의 init 과 겹치지 않게)
                    if (_reacquireXrWhileGated && t >= _minSettle
                        && !XRSettings.isDeviceActive && Time.unscaledTime >= nextXrTry)
                    {
                        nextXrTry = Time.unscaledTime + Mathf.Max(0.5f, _xrRetryInterval);
                        yield return TryStartXr();
                    }
                }
                else
                {
                    // 기본(타임드) 동작: 초기화 유예 후 활성 HMD가 전혀 없거나, 대기 한도를 넘기면 데스크톱으로.
                    if (t >= _minSettle && !XRSettings.isDeviceActive) break;
                    if (t >= _settleTimeout) break;
                }

                t += Time.unscaledDeltaTime;
                yield return null;
            }

            LockMode(DecideNow());
        }

        private AppMode DecideNow()
        {
            if (_override == ModeOverride.ForceVR) return AppMode.VR;
            if (_override == ModeOverride.ForceDesktop) return AppMode.DesktopKBM;
            return IsHmdWorn() ? AppMode.VR : AppMode.DesktopKBM;
        }

        private void LockMode(AppMode mode)
        {
            Mode = mode;
            Resolved = true;

            if (mode == AppMode.DesktopKBM && _stopXrInDesktop)
                StopXr();

            Debug.Log($"[XRModeManager] 실행 모드 확정: {mode}");
            ModeResolved?.Invoke(mode);
        }

        // HMD가 활성 + 착용(userPresence)되어 있는지. XR 미초기화/미연결이면 false.
        private static bool IsHmdWorn()
        {
            if (!XRSettings.isDeviceActive) return false;

            var head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            if (!head.isValid) return false;

            if (head.TryGetFeatureValue(CommonUsages.userPresence, out bool worn))
                return worn;

            // userPresence 미지원 런타임: 유효한 HMD가 잡히면 착용으로 간주.
            return true;
        }

        // 데스크톱 확정 시 XR 런타임을 완전히 내려 평면 렌더링으로 전환.
        private static void StopXr()
        {
            var mgr = XRGeneralSettings.Instance != null ? XRGeneralSettings.Instance.Manager : null;
            if (mgr == null || !mgr.isInitializationComplete) return;

            mgr.StopSubsystems();
            mgr.DeinitializeLoader();
        }

        // 대기 중 XR 재획득: 로더가 안 떠 있으면(기기 미연결·Link 끊김) 초기화를 1회 시도한다.
        // 성공하면 서브시스템을 시작해 XRSettings.isDeviceActive 가 활성화되고, 이후 착용 시 IsHmdWorn 이 VR을 잡는다.
        // 실패(기기 없음)면 activeLoader 가 null 로 남아 다음 주기에 다시 시도된다.
        private static IEnumerator TryStartXr()
        {
            var mgr = XRGeneralSettings.Instance != null ? XRGeneralSettings.Instance.Manager : null;
            if (mgr == null || mgr.activeLoader != null) yield break;   // 매니저 없음 or 이미 초기화됨 → 중복 init 방지

            yield return mgr.InitializeLoader();   // 비동기. 기기 없으면 activeLoader 는 null 로 유지된다.
            if (mgr.activeLoader != null)
                mgr.StartSubsystems();
        }
    }
}
