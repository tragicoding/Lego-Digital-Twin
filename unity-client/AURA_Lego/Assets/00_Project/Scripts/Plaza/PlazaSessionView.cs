using UnityEngine;
using TMPro;
using LegoTwin.Character;
using LegoTwin.Data;
using LegoTwin.UI;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// 광장 내 세션 하나의 시각적 표현.
    /// 캐릭터+오브제 위치에 배치되며 좋아요 수, 말풍선, 별 표시를 담당.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] likesCountText 에 TextMeshProUGUI 연결 (Canvas 안의 UI 텍스트)
    ///   [ ] bubbleText 에 TextMeshProUGUI 연결
    ///   [ ] starObject 에 별 표시 GameObject 연결 (1위 표시)
    ///   [ ] LikeSystem 컴포넌트가 같은 GameObject에 있어야 함
    /// </summary>
    public class PlazaSessionView : MonoBehaviour
    {
        public string SessionId { get; private set; }

        [Header("UI 연결")]
        [Tooltip("좋아요 수 텍스트 — Canvas 안의 TextMeshProUGUI 연결")]
        public TextMeshProUGUI likesCountText;

        [Tooltip("말풍선 텍스트 — Canvas 안의 TextMeshProUGUI 연결")]
        public TextMeshProUGUI bubbleText;

        [Tooltip("좋아요 1위 별 표시 오브젝트")]
        public GameObject starObject;

        [Tooltip("오브제 앞에 띄울 말풍선 패널(BubblePanel). 비우면 뷰 안에 그대로 둠.")]
        public RectTransform bubblePanel;

        private LikeSystem                _likeSystem;
        private Camera                    _cam;
        private PlacedCharacterController _placedCharacter;

        // ════════════════════════════════════════════════════════════
        // 초기화
        // ════════════════════════════════════════════════════════════

        private void Awake()
        {
            _cam = Camera.main;
        }

        private void Start()
        {
            // Awake보다 Start에서 Camera.main이 더 안정적으로 초기화됨
            if (_cam == null) _cam = Camera.main;

            // World Space Canvas는 Event Camera가 설정되어야 마우스 클릭이 동작함
            foreach (var canvas in GetComponentsInChildren<Canvas>(includeInactive: true))
            {
                if (canvas.renderMode == RenderMode.WorldSpace && canvas.worldCamera == null)
                    canvas.worldCamera = _cam;
            }
        }

        private void LateUpdate()
        {
            // 카메라 방향과 평행(edge-on)이 되면 GraphicRaycaster 교점 계산이 NaN → 빌보드로 방지
            if (_cam == null) _cam = Camera.main;
            if (_cam != null)
                transform.forward = _cam.transform.forward;
        }

/// <summary>PlazaManager가 세션 생성 시 호출.</summary>
        public void Initialize(PlazaSessionData session, bool isTop)
        {
            SessionId = session.session_id;

            UpdateLikes(session.likes);
            SetTopLiked(isTop);

            if (bubbleText != null)
                bubbleText.text = session.bubble_text ?? "";

            // LikeSystem 연결
            _likeSystem = GetComponent<LikeSystem>();
            if (_likeSystem != null)
                _likeSystem.Initialize(session.session_id, session.likes);
        }

        // ════════════════════════════════════════════════════════════
        // 좋아요 갱신 (WebSocket 또는 폴링에서 호출)
        // ════════════════════════════════════════════════════════════

        public void UpdateLikes(int likes)
        {
            if (likesCountText != null)
                likesCountText.text = $"♥ {likes}";
            _likeSystem?.UpdateCount(likes);
        }

        public void SetTopLiked(bool isTop)
        {
            if (starObject == null) return;

            starObject.SetActive(isTop);
            var heartEffect = starObject.GetComponent<HeartEffect>();

            if (isTop)
                // SetActive(true) 직후 같은 프레임에서 StartCoroutine 하면 inactive 에러 발생
                // → PlazaSessionView(항상 활성)에서 한 프레임 대기 후 PlayLoop 호출
                StartCoroutine(PlayLoopNextFrame(heartEffect));
            else
                heartEffect?.StopLoop();
        }

        private System.Collections.IEnumerator PlayLoopNextFrame(HeartEffect heartEffect)
        {
            yield return null;
            heartEffect?.PlayLoop();
        }

        // ════════════════════════════════════════════════════════════
        // 시그니처 동작 — LikeSystem이 접근/이탈 시 호출
        // ════════════════════════════════════════════════════════════

        /// <summary>PlazaManager가 스폰 직후 호출해 배치 캐릭터를 연결한다.</summary>
        public void SetCharacter(PlacedCharacterController character) => _placedCharacter = character;

        public void PlaySignatureMotion()  => _placedCharacter?.PlaySignatureMotion();
        public void StopSignatureMotion()  => _placedCharacter?.StopSignatureMotion();

        // ════════════════════════════════════════════════════════════
        // BubblePanel(말풍선) — 오브제 위에 분리 배치
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// BubblePanel(말풍선)을 뷰에서 분리해 오브제 위에 떠 있도록 재배치한다.
        /// 카메라를 향해 빌보드되며 오브제를 따라간다(오브제 파괴 시 자동 정리 — NameTag 재사용).
        /// bubblePanel 또는 objectTransform 이 없으면 아무것도 하지 않고 뷰 안에 그대로 둔다.
        /// </summary>
        public void PlaceBubbleAtObject(Transform objectTransform, float heightOffset,
                                        float scaleMultiplier = 1f, float textScale = 1f,
                                        float rightOffset = 0f)
        {
            if (bubblePanel == null || objectTransform == null) return;

            // BubblePanel 은 stretch 앵커(부모 캔버스 크기에 맞춤)라, 재부모 시 크기 기준이 깨지면
            // 패널·텍스트가 어긋난다. 원래 패널 크기·월드 스케일을 캡처해 그대로 복원한다.
            Canvas.ForceUpdateCanvases();                  // rect 계산 보장
            Vector2 panelSize = bubblePanel.rect.size;
            if (panelSize.x < 1f || panelSize.y < 1f) panelSize = new Vector2(800f, 800f); // 폴백
            Vector3 panelWorldScale = bubblePanel.lossyScale;
            float s = Mathf.Max(0.01f, scaleMultiplier);

            // 오브제를 따라다닐 독립 World Space 캔버스 (뷰의 빌보드와 분리)
            var anchor = new GameObject("BubbleAnchor", typeof(RectTransform), typeof(Canvas));
            var canvas = anchor.GetComponent<Canvas>();
            canvas.renderMode  = RenderMode.WorldSpace;
            canvas.worldCamera = _cam != null ? _cam : Camera.main;
            var art = (RectTransform)anchor.transform;
            art.sizeDelta  = panelSize;            // 패널 원래 크기로 캔버스 크기 지정(stretch 기준 복원)
            art.localScale = panelWorldScale * s;  // 원래 월드 스케일 × 배율

            // 패널을 캔버스에 꽉 채워 중앙 정렬 → 프리팹의 offset(400,400)/z(300) quirk 제거.
            // BubbleText 는 패널의 stretch 자식이라 패널이 제자리면 자동으로 중앙에 온다.
            bubblePanel.SetParent(anchor.transform, worldPositionStays: false);
            bubblePanel.anchorMin = Vector2.zero;
            bubblePanel.anchorMax = Vector2.one;
            bubblePanel.offsetMin = Vector2.zero;
            bubblePanel.offsetMax = Vector2.zero;
            bubblePanel.localScale = Vector3.one;
            var lp = bubblePanel.localPosition; lp.z = 0f; bubblePanel.localPosition = lp;

            // 글씨 크기만 독립 조정(위치는 패널 중앙 유지) — fontSize 로, 패널 배율 상쇄.
            if (bubbleText != null)
            {
                bubbleText.transform.localScale = Vector3.one;
                bubbleText.enableAutoSizing = false;
                bubbleText.fontSize *= textScale / s;
            }

            // 오브제 머리 위 추적 + 카메라 빌보드 — NameTag 의 위치/빌보드 로직 재사용
            anchor.AddComponent<NameTag>().Bind(objectTransform, heightOffset, rightOffset);
        }
    }
}
