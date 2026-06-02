using System.Collections;
using TMPro;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// 배회 NPC 머리 위 말풍선 UI.
    ///
    /// 씬 Hierarchy 구조 (NPC 자식으로 배치):
    ///   NPC_01
    ///     └── SpeechBubbleCanvas    Canvas (World Space) + NPCSpeechBubble
    ///          └── BubbleRoot       GameObject + CanvasGroup
    ///               └── BubblePanel Image
    ///                    └── BubbleText  TextMeshProUGUI
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] NPC 자식으로 Canvas 생성 → Render Mode: World Space
    ///   [ ] Canvas Transform: NPC 머리 위 위치로 조정 (Y 오프셋)
    ///   [ ] Canvas Scale: (0.01, 0.01, 0.01) 등 씬 스케일에 맞게
    ///   [ ] 이 스크립트를 Canvas GameObject에 추가
    ///   [ ] _bubbleText, _canvasGroup Inspector 연결
    ///   [ ] _messages 배열에 대사 입력
    /// </summary>
    public class NPCSpeechBubble : MonoBehaviour
    {
        [Header("UI 연결")]
        [SerializeField] private TMP_Text    _bubbleText;
        [SerializeField] private CanvasGroup _canvasGroup;

        [Header("대사")]
        [Tooltip("표시할 대사 목록. 여러 개면 순서대로 순환.")]
        [SerializeField] private string[] _messages = { "안녕하세요!" };

        [Header("연출 설정")]
        [Tooltip("한 대사 표시 시간 (초)")]
        [SerializeField] private float _displayDuration = 4f;

        [Tooltip("대사 전환 페이드 시간 (초)")]
        [SerializeField] private float _fadeDuration = 0.4f;

        [Tooltip("대사 없이 숨어있는 시간 (초). 0이면 항상 표시.")]
        [SerializeField] private float _hideDuration = 2f;

        private int       _currentIndex = 0;
        private Camera    _cam;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Start()
        {
            _cam = Camera.main;

            // World Space Canvas Event Camera 자동 설정
            var canvas = GetComponent<Canvas>();
            if (canvas != null && canvas.renderMode == RenderMode.WorldSpace && canvas.worldCamera == null)
                canvas.worldCamera = _cam;

            if (_canvasGroup != null) _canvasGroup.alpha = 0f;

            if (_messages != null && _messages.Length > 0)
                StartCoroutine(BubbleRoutine());
        }

        private void LateUpdate()
        {
            if (_cam == null) _cam = Camera.main;
            if (_cam != null)
                transform.forward = _cam.transform.forward;
        }

        // ════════════════════════════════════════════════════════════
        // 말풍선 루틴
        // ════════════════════════════════════════════════════════════

        private IEnumerator BubbleRoutine()
        {
            while (true)
            {
                // 대사 설정
                if (_bubbleText != null)
                    _bubbleText.text = _messages[_currentIndex];

                // 페이드 인
                yield return FadeRoutine(1f);

                // 표시 유지
                yield return new WaitForSeconds(_displayDuration);

                // 페이드 아웃
                yield return FadeRoutine(0f);

                // 숨김 대기
                if (_hideDuration > 0f)
                    yield return new WaitForSeconds(_hideDuration);

                // 다음 대사로 순환
                _currentIndex = (_currentIndex + 1) % _messages.Length;
            }
        }

        private IEnumerator FadeRoutine(float target)
        {
            if (_canvasGroup == null) yield break;
            float start = _canvasGroup.alpha, elapsed = 0f;
            while (elapsed < _fadeDuration)
            {
                elapsed += Time.deltaTime;
                _canvasGroup.alpha = Mathf.Lerp(start, target, elapsed / _fadeDuration);
                yield return null;
            }
            _canvasGroup.alpha = target;
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>대사를 즉시 변경한다.</summary>
        public void SetMessage(string message)
        {
            if (_bubbleText != null) _bubbleText.text = message;
        }
    }
}
