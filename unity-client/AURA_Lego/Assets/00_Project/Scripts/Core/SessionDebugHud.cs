using UnityEngine;
using LegoTwin.Data;
using LegoTwin.Managers;
using LegoTwin.Plaza;
using LegoTwin.Character;

namespace LegoTwin.Core
{
    /// <summary>
    /// (디버그 전용) 세션 생명주기 검증 HUD. 빈 GameObject에 부착 후 Play.
    /// 좌상단에 현재 세션·리셋 상태를 실시간 표시하고, 단축키로 종료 사이클을 즉시 테스트한다.
    /// Mock·Server 동일하게 동작한다(읽는 값이 전부 모드 무관). 전시 빌드에서는 _enabled를 끄거나
    /// 이 GameObject를 비활성화한다.
    ///
    /// 검증 포인트(A안 종료→대기화면 복귀 + static 리셋):
    ///   1) [End] 키로 EndCurrentSession 호출 → 씬 리로드 → 대기화면 → 다음 세션
    ///   2) 새 세션 진입 시 IsGuideMode 가 true 로 복귀하는지
    ///   3) 좋아요를 누른 뒤 종료→다음 세션에서 "투표 기록"이 0 으로 리셋되는지
    ///   4) "캐시 머티리얼 세트"가 세션을 거듭해도 누적되지 않는지(누수 방지)
    ///
    /// 사용법:
    ///   - 종료 사이클 테스트: [End] (또는 _endSessionKey)
    ///   - 좋아요 리셋 테스트: 광장에서 하트 → "투표 기록" 증가 확인 → [End] → 다음 세션에서 0 확인
    /// </summary>
    public class SessionDebugHud : MonoBehaviour
    {
        [Tooltip("HUD 표시 및 단축키 활성화. 전시 빌드에서는 해제.")]
        [SerializeField] private bool _enabled = true;

        [Tooltip("현재 세션을 종료하고 대기화면으로 복귀시키는 테스트 키 (QuitButton과 동일 동작).")]
        [SerializeField] private KeyCode _endSessionKey = KeyCode.End;

        // 씬 리로드로 이 컴포넌트는 매번 새로 생성된다. 누적 사이클 수는 static으로 유지해
        // 종료→리로드→재진입이 실제로 반복되는지 보이게 한다(리셋 검증과 별개).
        private static int _sessionCycleCount;

        private void Update()
        {
            if (!_enabled) return;
            if (Input.GetKeyDown(_endSessionKey))
            {
                _sessionCycleCount++;
                SessionManager.Instance?.EndCurrentSession();
            }
        }

        private void OnGUI()
        {
            if (!_enabled) return;

            var sm = SessionManager.Instance;
            SessionData s = sm != null ? sm.CurrentSession : null;

            string mode    = sm != null ? sm.dataSourceMode.ToString() : "(SessionManager 없음)";
            string session = s != null ? $"{s.character_npc_name} / {s.session_id}" : "(로드 대기 중…)";

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("── 세션 검증 HUD ──");
            sb.AppendLine($"모드            : {mode}");
            sb.AppendLine($"현재 세션       : {session}");
            sb.AppendLine($"가이드 모드     : {GameFlowManager.IsGuideMode}   (새 세션=True 여야 정상)");
            sb.AppendLine($"투표 기록 수    : {LikeSystem.VotedSessionCount}   (새 세션=0 이어야 정상)");
            sb.AppendLine($"캐시 머티리얼   : {CharacterAnimationController.CachedMaterialSetCount}   (세션 거듭해도 누적 X)");
            sb.AppendLine($"종료 사이클 수  : {_sessionCycleCount}");
            sb.AppendLine($"[{_endSessionKey}] 종료→대기화면 복귀 테스트");

            var style = new GUIStyle(GUI.skin.box)
            {
                alignment = TextAnchor.UpperLeft,
                fontSize  = 14,
                padding   = new RectOffset(10, 10, 10, 10),
            };
            style.normal.textColor = Color.white;

            GUI.Box(new Rect(10, 10, 420, 168), sb.ToString(), style);
        }
    }
}
