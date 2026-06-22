using UnityEngine;
using UnityEngine.EventSystems;

namespace LegoTwin.Core
{
    /// <summary>
    /// EventSystem 의 UI 입력 모듈을 실행 모드(VR/데스크톱)에 맞춰 한 번 토글한다.
    ///
    /// 저결합·확장성
    ///   - 특정 모듈 타입에 의존하지 않고 Behaviour 배열로 받는다.
    ///     (예: VR=XR UI Input Module, 데스크톱=InputSystemUIInputModule —
    ///      구성이 바뀌어도 인스펙터 연결만 바꾸면 됨)
    ///   - XRModeManager 의 정적 상태만 읽는다(이벤트 누수 없음). 매니저가 없으면
    ///     아무것도 바꾸지 않아 씬 기본 모듈이 그대로 유지된다(무회귀).
    ///
    /// 사용: EventSystem 오브젝트에 부착 후 _vrModules / _desktopModules 연결.
    /// </summary>
    [RequireComponent(typeof(EventSystem))]
    public class EventSystemModeBinder : MonoBehaviour
    {
        [Tooltip("VR 모드에서 켤 입력 모듈/컴포넌트 (데스크톱에선 끔)")]
        [SerializeField] private Behaviour[] _vrModules;

        [Tooltip("데스크톱(키마) 모드에서 켤 입력 모듈/컴포넌트 (VR에선 끔)")]
        [SerializeField] private Behaviour[] _desktopModules;

        private void Update()
        {
            if (!XRModeManager.Resolved) return;   // 확정 전엔 씬 기본 모듈 유지
            Apply(XRModeManager.Mode);
            enabled = false;                       // 1회 적용 후 정지
        }

        private void Apply(AppMode mode)
        {
            bool vr = mode == AppMode.VR;
            SetEnabled(_vrModules, vr);
            SetEnabled(_desktopModules, !vr);
        }

        private static void SetEnabled(Behaviour[] modules, bool on)
        {
            if (modules == null) return;
            foreach (var m in modules)
                if (m != null) m.enabled = on;
        }
    }
}
