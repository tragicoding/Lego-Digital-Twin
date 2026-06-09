using System;
using TMPro;
using UnityEngine;
using LegoTwin.Managers;

namespace LegoTwin.UI
{
    /// <summary>
    /// bubble_text(오브제 한마디) 입력 흐름의 단일 진입점.
    ///
    /// 가이드 모드(GameFlowManager)와 자유 모드(OwnCreationInteractor)가 모두 이 헬퍼를 사용한다.
    /// 저장은 항상 SessionManager.SetBubbleText 한 곳만 거치므로
    /// Mock / Server 모드가 호출부 코드 분기 없이 동일하게 동작한다.
    ///   - Mock   : CurrentSession 메모리에만 반영 (앱 종료 시 PlazaManager가 파일로 저장)
    ///   - Server : 메모리 반영 + PATCH /sessions/{id}/profile 즉시 전송
    ///
    /// 입력 UI는 MotionPromptUI를 재사용한다(별도 입력창 불필요).
    /// </summary>
    public static class BubbleTextPrompt
    {
        public const string DefaultTitle = "오브제에 남길 \n한마디를 입력하세요";

        /// <summary>
        /// 입력창을 열어 bubble_text 편집을 시작한다.
        /// 제출 시 SessionManager.SetBubbleText로 저장하고, liveLabel이 있으면 즉시 갱신한 뒤
        /// onDone(저장된 값)을 호출한다. promptUI가 없으면 경고 후 onDone(null)으로 흐름을 진행시킨다.
        /// </summary>
        /// <param name="promptUI">재사용할 입력 UI (가이드/자유 모드가 공유하는 동일 컴포넌트)</param>
        /// <param name="onDone">완료 콜백(저장된 값). 시나리오/인터랙션 진행 신호로 사용</param>
        /// <param name="liveLabel">즉시 갱신할 말풍선 TMP. 없으면 생략(가이드 모드처럼 뷰가 아직 없을 때)</param>
        /// <param name="title">입력창 제목(미지정 시 DefaultTitle)</param>
        public static void Open(
            MotionPromptUI promptUI,
            Action<string> onDone = null,
            TMP_Text liveLabel = null,
            string title = null)
        {
            if (promptUI == null)
            {
                Debug.LogWarning("[BubbleTextPrompt] MotionPromptUI 미연결 — bubble_text 입력 생략");
                onDone?.Invoke(null);
                return;
            }

            promptUI.Show(
                input =>
                {
                    // MotionPromptUI는 빈 입력을 막지만, 방어적으로 기존 값으로 폴백한다.
                    string value = string.IsNullOrWhiteSpace(input) ? CurrentValue(liveLabel) : input.Trim();
                    if (!string.IsNullOrEmpty(value))
                    {
                        SessionManager.Instance?.SetBubbleText(value);   // 단일 저장 경로 (Mock/Server 공통)
                        if (liveLabel != null) liveLabel.text = value;   // UX: 화면 즉시 반영
                    }
                    onDone?.Invoke(value);
                },
                string.IsNullOrEmpty(title) ? DefaultTitle : title,
                CurrentValue(liveLabel));   // 기존 값 prefill (수정 가능)
        }

        // 현재 bubble_text — 세션 메모리 우선, 없으면 화면 라벨 폴백.
        private static string CurrentValue(TMP_Text liveLabel)
        {
            var session = SessionManager.Instance?.CurrentSession;
            if (!string.IsNullOrWhiteSpace(session?.bubble_text))
                return session.bubble_text.Trim();
            if (liveLabel != null)
                return liveLabel.text ?? string.Empty;
            return string.Empty;
        }
    }
}
