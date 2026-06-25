using UnityEngine;

namespace LegoTwin.Core
{
    /// <summary>
    /// 운영자 전용 "진짜 앱 종료" — 전시 중 Unity 수정/재빌드가 필요할 때 사용한다.
    /// (씬의 GameFlowManager 오브젝트 등에 이 컴포넌트를 추가해서 쓴다.)
    ///
    /// 관람객용 종료 버튼(QuitButton)과 '분리된' 동작이다:
    ///   - QuitButton  → SessionManager.EndCurrentSession() : 현재 세션을 '완료' 처리하고 큐를 다음으로 넘긴 뒤
    ///                                                         씬 리로드 → 대기화면 복귀(다음 관람객).
    ///   - OperatorQuit → Application.Quit()                : 큐/세션을 건드리지 않고 앱만 종료.
    ///
    /// 재시작 시 동작(Server 모드):
    ///   OperatorQuit / 크래시 / 정전 등 어떤 방식으로 종료해도 재시작하면 '다음 세션'으로 넘어간다.
    ///   - SessionManager.OnApplicationQuit()이 best-effort로 AdvanceQueue를 시도한다.
    ///   - 실패하더라도 서버 GET /sessions/active 가 "runtime" 상태 세션을 스테일로 판단해
    ///     자동 advance 후 다음 큐를 팝한다.
    ///   (Mock 모드는 큐/세션을 건드리지 않으므로 재시작해도 영향 없음.)
    ///
    /// 기본 종료 키는 Esc. 관람객 오발이 걱정되면 보조키(Ctrl/Shift) 요구를 켤 수 있다.
    /// 의존성 0 — SessionManager/큐를 직접 참조하지 않는다.
    /// Mock·Server 어느 모드든 동일하게 동작한다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] 씬의 GameFlowManager 오브젝트(또는 아무 GameObject)에 이 컴포넌트 추가
    ///   [ ] 필요 시 종료 키/보조키 변경
    /// </summary>
    public class OperatorQuit : MonoBehaviour
    {
        [Header("운영자 종료 키")]
        [Tooltip("이 키를 누르면 앱 종료 (기본 Esc)")]
        [SerializeField] private KeyCode _quitKey = KeyCode.Escape;

        [Header("관람객 오발 방지 (선택)")]
        [Tooltip("켜면 Ctrl을 누른 채여야 종료")]
        [SerializeField] private bool _requireCtrl  = false;
        [Tooltip("켜면 Shift를 누른 채여야 종료")]
        [SerializeField] private bool _requireShift = false;

        private void Update()
        {
            if (_requireCtrl  && !(Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl))) return;
            if (_requireShift && !(Input.GetKey(KeyCode.LeftShift)   || Input.GetKey(KeyCode.RightShift)))   return;
            if (Input.GetKeyDown(_quitKey)) QuitApp();
        }

        /// <summary>앱 종료(운영자용). 재시작 시 다음 세션으로 넘어간다.</summary>
        public void QuitApp()
        {
            Debug.Log("[OperatorQuit] 운영자 종료 — 큐/세션 보존(재시작 시 같은 세션부터 이어짐)");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
}
