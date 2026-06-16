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
    /// 재시작 시 이어가기(핵심):
    ///   운영자 종료는 '완료 처리(advance)'를 하지 않으므로 큐가 그대로 남는다.
    ///   재빌드된 앱을 다시 켜면 마지막 큐의 '같은 세션'부터 이어진다.
    ///     - Mock  : MockSessionLoader 인덱스(PlayerPrefs) 유지 → Load()가 같은 세션 반환
    ///     - Server: 백엔드 큐 그대로 → GET /sessions/active 가 같은 세션 반환
    ///   (크래시/정전 등 비정상 종료에도 동일하게 같은 세션부터 이어진다 — 완료 시에만 advance하므로.)
    ///   ※ 세션은 '대기화면부터' 다시 시작된다(체험 중간 지점 복원은 아님 — 큐/세션 단위 이어가기).
    ///
    /// 기본 종료 키는 Esc. 관람객 오발이 걱정되면 보조키(Ctrl/Shift) 요구를 켤 수 있다.
    /// 의존성 0 — SessionManager/큐를 참조하지 않는다(advance를 '안 하는' 것이 이어가기의 핵심이라 추가 작업 불필요).
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

        /// <summary>큐/세션을 advance 하지 않고 앱만 종료(운영자용). 재시작 시 같은 세션부터 이어진다.</summary>
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
