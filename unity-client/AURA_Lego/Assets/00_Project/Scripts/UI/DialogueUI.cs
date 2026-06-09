using System.Collections;
using TMPro;
using UnityEngine;

namespace LegoTwin.UI
{
    /// <summary>
    /// 가이드 NPC 말풍선 UI — World Space Canvas.
    ///
    /// 역할:
    ///   - GameFlowManager.OnDialogueChanged 이벤트를 받아 텍스트 표시
    ///   - _followTarget(NPC transform)이 설정되면 머리 위 위치를 매 프레임 추적
    ///   - Billboard 모드: 항상 카메라 방향으로 회전
    ///   - 페이드 인/아웃 연출
    ///
    /// 씬 Hierarchy 구조:
    ///   DialogueUI              (Canvas — World Space, Scale ~0.01)
    ///     └── BubbleRoot        (GameObject — Show/Hide 대상, CanvasGroup)
    ///          └── BubblePanel  (Image — 배경 패널)
    ///               ├── SpeakerNameText  (TMP_Text — NPC 이름)
    ///               └── DialogueText     (TMP_Text — 대사 내용)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Canvas: Render Mode → World Space
    ///   [ ] Canvas: Event Camera → Main Camera 연결
    ///   [ ] Canvas Transform: Scale → (0.01, 0.01, 0.01) 또는 원하는 크기로 조정
    ///   [ ] DialogueUI: Billboard Mode ☑ 체크
    ///   [ ] DialogueUI: Height Offset — NPC 스케일에 맞게 조정 (기본 2.2)
    ///   [ ] BubbleRoot / SpeakerNameText / DialogueText / CanvasGroup 연결
    ///   [ ] GameFlowManager._dialogueUI 에 이 GameObject 드래그
    ///   ※ SetFollowTarget()은 NPC 스폰 시 GameFlowManager가 자동으로 호출함
    /// </summary>
    public class DialogueUI : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("UI 요소")]
        [Tooltip("Show/Hide 할 루트 오브젝트 (BubbleRoot)")]
        [SerializeField] private GameObject  _bubbleRoot;

        [Tooltip("화자 이름 텍스트 (NPC 이름 표시, 없으면 생략)")]
        [SerializeField] private TMP_Text    _speakerNameText;

        [Tooltip("대사 내용 텍스트")]
        [SerializeField] private TMP_Text    _dialogueText;

        [Tooltip("페이드 인/아웃용 CanvasGroup (BubbleRoot 에 추가)")]
        [SerializeField] private CanvasGroup _canvasGroup;

        [Header("연출 설정")]
        [Tooltip("페이드 인/아웃 시간 (초)")]
        [SerializeField] private float _fadeDuration = 0.25f;

        [Tooltip("World Space Canvas 사용 시 true — 항상 카메라 방향으로 회전")]
        [SerializeField] private bool  _billboardMode = true;

        [Header("NPC 말풍선 추적")]
        [Tooltip("말풍선이 따라갈 NPC Transform. GameFlowManager가 스폰 후 SetFollowTarget()으로 주입.")]
        [SerializeField] private Transform _followTarget;

        [Tooltip("_followTarget 기준 머리 위 높이 오프셋 (NPC 스케일에 맞게 조정)")]
        [SerializeField] private float _heightOffset = 2.2f;

        [Tooltip("NPC 오른쪽 방향 오프셋 (NPC의 로컬 right 기준, 음수면 왼쪽)")]
        [SerializeField] private float _rightOffset = 0f;

        // ── 런타임 ───────────────────────────────────────────────────
        private Coroutine _fadeCoroutine;
        private Camera    _mainCamera;

        // ════════════════════════════════════════════════════════════
        // Unity 생명주기
        // ════════════════════════════════════════════════════════════

        private void Awake()
        {
            _mainCamera = Camera.main;

            // 시작 시 숨김
            if (_bubbleRoot != null)
                _bubbleRoot.SetActive(false);

            if (_canvasGroup != null)
                _canvasGroup.alpha = 0f;
        }

        private void LateUpdate()
        {
            // NPC 머리 위 + 오른쪽 오프셋 추적
            if (_followTarget != null)
                transform.position = _followTarget.position
                    + Vector3.up          * _heightOffset
                    + _followTarget.right * _rightOffset;

            // Billboard: 카메라와 같은 방향으로 정렬 (World Space Canvas 표준)
            if (!_billboardMode) return;
            if (_mainCamera == null) _mainCamera = Camera.main;
            if (_mainCamera != null)
                transform.forward = _mainCamera.transform.forward;
        }

        // ════════════════════════════════════════════════════════════
        // 공개 API — GameFlowManager 에서 호출
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// NPC 스폰 후 말풍선이 따라갈 타겟을 설정한다.
        /// GameFlowManager.OnSessionLoaded 에서 SpawnGuide 직후 호출.
        /// </summary>
        public void SetFollowTarget(Transform target) => _followTarget = target;

        /// <summary>
        /// 말풍선을 표시한다. 이미 표시 중이면 텍스트만 교체한다.
        /// </summary>
        /// <param name="speakerName">화자 이름 (NPC 이름)</param>
        /// <param name="text">대사 내용</param>
        public void Show(string speakerName, string text)
        {
            // 텍스트 설정
            if (_speakerNameText != null)
                _speakerNameText.text = speakerName;

            if (_dialogueText != null)
                _dialogueText.text = text;

            // 루트 활성화 후 페이드 인
            if (_bubbleRoot != null)
                _bubbleRoot.SetActive(true);

            FadeTo(1f);
        }

        /// <summary>
        /// 말풍선을 숨긴다 (페이드 아웃 후 비활성화).
        /// </summary>
        public void Hide()
        {
            if (_fadeCoroutine != null)
                StopCoroutine(_fadeCoroutine);

            _fadeCoroutine = StartCoroutine(HideRoutine());
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private void FadeTo(float target)
        {
            if (_fadeCoroutine != null)
                StopCoroutine(_fadeCoroutine);

            _fadeCoroutine = StartCoroutine(FadeRoutine(target));
        }

        private IEnumerator FadeRoutine(float target)
        {
            if (_canvasGroup == null)
            {
                yield break;
            }

            float start   = _canvasGroup.alpha;
            float elapsed = 0f;

            while (elapsed < _fadeDuration)
            {
                elapsed           += Time.deltaTime;
                _canvasGroup.alpha = Mathf.Lerp(start, target, elapsed / _fadeDuration);
                yield return null;
            }

            _canvasGroup.alpha = target;
            _fadeCoroutine     = null;
        }

        private IEnumerator HideRoutine()
        {
            // 페이드 아웃
            yield return FadeRoutine(0f);

            // 완전히 투명해진 뒤 비활성화
            if (_bubbleRoot != null)
                _bubbleRoot.SetActive(false);
        }
    }
}
