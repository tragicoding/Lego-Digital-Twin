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
        // 오브제 위 패널 배치 — BubblePanel(말풍선) / LikePanel(좋아요)
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

            ReparentPanelToObject(bubblePanel, objectTransform, heightOffset, scaleMultiplier, rightOffset);

            // 글씨 크기만 독립 조정(위치는 패널 중앙 유지) — fontSize 로, 패널 배율 상쇄.
            if (bubbleText != null)
            {
                float s = Mathf.Max(0.01f, scaleMultiplier);
                bubbleText.transform.localScale = Vector3.one;
                bubbleText.enableAutoSizing = false;
                bubbleText.fontSize *= textScale / s;
            }
        }

        /// <summary>
        /// LikePanel(좋아요 패널)을 오브제 위에 떠 있도록 재배치한다(카메라 빌보드, 오브제 추적).
        /// LikeSystem.likePanel 을 사용한다. 클릭 감지는 LikeSystem 이 카메라 기반으로 직접
        /// 처리하므로(GraphicRaycaster 불필요) 재배치 후에도 그대로 동작한다.
        /// likePanel 또는 objectTransform 이 없으면 아무것도 하지 않는다(예: 자기 창작물 — 투표 불가).
        /// </summary>
        public void PlaceLikePanelAtObject(Transform objectTransform, float heightOffset,
                                           float scaleMultiplier = 1f, float rightOffset = 0f,
                                           float forwardOffset = 0f, float forwardSizeFactor = 0f)
        {
            if (objectTransform == null) return;
            var likeSystem = _likeSystem != null ? _likeSystem : GetComponent<LikeSystem>();
            var panel = (likeSystem != null && likeSystem.likePanel != null)
                ? likeSystem.likePanel.transform as RectTransform : null;
            if (panel == null) return;   // 비어 있거나 RectTransform 아님(자기 창작물 등) → 무시

            // 좋아요 패널은 오브제 '앞'(카메라 쪽) + 플레이어 눈높이에 — 크기 비례 전진.
            ReparentPanelToObject(panel, objectTransform, heightOffset, scaleMultiplier, rightOffset,
                                  forwardOffset, forwardSizeFactor: forwardSizeFactor, eyeLevel: true);
        }

        /// <summary>
        /// 패널(RectTransform)을 뷰의 빌보드와 분리해, 오브제를 따라다니는 독립 World Space 캔버스로
        /// 재배치한다(카메라 빌보드). 패널이 stretch 앵커(부모 캔버스 크기에 맞춤)라 재부모 시 기준이
        /// 깨지지 않도록 원래 크기·월드 스케일을 캡처해 복원하고, 캔버스를 꽉 채워 중앙 정렬한다.
        /// 비활성 패널(LikePanel 은 접근 전 비활성)도 정확히 측정하도록 잠시 활성화 후 원상복구한다.
        /// </summary>
        private GameObject ReparentPanelToObject(RectTransform panel, Transform objectTransform,
                                                 float heightOffset, float scaleMultiplier, float rightOffset,
                                                 float forwardOffset = 0f, float forwardSizeFactor = 0f,
                                                 bool eyeLevel = false)
        {
            bool wasActive = panel.gameObject.activeSelf;
            if (!wasActive) panel.gameObject.SetActive(true);   // 비활성이면 측정 위해 잠시 켬

            Canvas.ForceUpdateCanvases();                  // rect 계산 보장
            Vector2 panelSize = panel.rect.size;
            if (panelSize.x < 1f || panelSize.y < 1f) panelSize = new Vector2(800f, 800f); // 폴백
            Vector3 panelWorldScale = panel.lossyScale;
            float s = Mathf.Max(0.01f, scaleMultiplier);

            // 오브제를 따라다닐 독립 World Space 캔버스 (뷰의 빌보드와 분리)
            var anchor = new GameObject("PanelAnchor", typeof(RectTransform), typeof(Canvas));
            var canvas = anchor.GetComponent<Canvas>();
            canvas.renderMode  = RenderMode.WorldSpace;
            canvas.worldCamera = _cam != null ? _cam : Camera.main;
            var art = (RectTransform)anchor.transform;
            art.sizeDelta  = panelSize;            // 패널 원래 크기로 캔버스 크기 지정(stretch 기준 복원)
            art.localScale = panelWorldScale * s;  // 원래 월드 스케일 × 배율

            // 패널을 캔버스에 꽉 채워 중앙 정렬 → 프리팹의 offset/z quirk 제거.
            panel.SetParent(anchor.transform, worldPositionStays: false);
            panel.anchorMin = Vector2.zero;
            panel.anchorMax = Vector2.one;
            panel.offsetMin = Vector2.zero;
            panel.offsetMax = Vector2.zero;
            panel.localScale = Vector3.one;
            var lp = panel.localPosition; lp.z = 0f; panel.localPosition = lp;

            if (!wasActive) panel.gameObject.SetActive(false); // 원래 상태 복원(접근 시 LikeSystem 이 켬)

            // 오브제 추적 + 카메라 빌보드 — NameTag 의 위치/빌보드 로직 재사용
            anchor.AddComponent<NameTag>().Bind(objectTransform, heightOffset, rightOffset,
                                                forwardOffset, forwardSizeFactor, eyeLevel);
            return anchor;
        }
    }
}
