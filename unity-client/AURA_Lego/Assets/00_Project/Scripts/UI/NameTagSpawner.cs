using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace LegoTwin.UI
{
    /// <summary>
    /// 이름표 생성기(싱글톤). 캐릭터/오브제 스폰 합류점에서 Attach(target, name) 한 번만 호출하면
    /// 대상 위에 떠다니는 이름표(NameTag)를 만들어 준다.
    ///
    /// 저결합: 스포너들은 NameTagSpawner.Instance?.Attach(go, name) 로 '이름 문자열'만 넘긴다.
    ///         (UI 구성·폰트·스타일을 알 필요 없음. 씬에 NameTagSpawner 가 없으면 무해하게 무시됨.)
    /// 확장성: 라벨 모양(폰트·크기·배경·높이)은 이 컴포넌트의 Inspector 한 곳에서 조정.
    /// 서버 동일: Mock·Server 공통 합류점에서 호출되므로 데이터 출처와 무관하게 동일 동작.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] 씬에 빈 GameObject 하나 만들어 NameTagSpawner 추가
    ///   [ ] Font 에 한글 TMP 폰트 연결(비우면 TMP 기본 폰트 — 한글 깨질 수 있음)
    /// </summary>
    public class NameTagSpawner : MonoBehaviour
    {
        public static NameTagSpawner Instance { get; private set; }

        [Header("폰트 / 스타일")]
        [Tooltip("이름표 TMP 폰트(한글 지원 권장). 비우면 TMP 기본 폰트 사용.")]
        [SerializeField] private TMP_FontAsset _font;
        [SerializeField] private float _fontSize        = 36f;
        [SerializeField] private Color _textColor       = Color.white;
        [SerializeField] private bool  _background      = true;
        [SerializeField] private Color _backgroundColor = new Color(0f, 0f, 0f, 0.5f);

        [Header("배치")]
        [Tooltip("캐릭터 이름표를 머리(바운즈 상단)에서 위로 띄울 높이 (m)")]
        [SerializeField] private float _heightOffset = 0.5f;
        [Tooltip("오브제 이름표를 상단에서 위로 띄울 높이 (m) — 캐릭터와 별도")]
        [SerializeField] private float _objectHeightOffset = 0.5f;
        [Tooltip("World Space 캔버스 스케일 (오브제 기준)")]
        [SerializeField] private float _worldScale = 0.02f;
        [Tooltip("캐릭터 이름표 크기 배율 (오브제 대비). 1보다 작으면 더 작게.")]
        [SerializeField] private float _characterScaleMultiplier = 0.6f;
        [Tooltip("이름표 캔버스 크기 (px)")]
        [SerializeField] private Vector2 _size = new Vector2(400f, 90f);

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(this); return; }
            Instance = this;
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        /// <summary>오브제용 이름표(기본 크기·오브제 높이). 캐릭터는 AttachCharacter 사용.</summary>
        public void Attach(GameObject target, string name)
            => Build(target, name, _worldScale, _objectHeightOffset);

        /// <summary>캐릭터용 이름표 — 오브제보다 _characterScaleMultiplier 배만큼 작게(캐릭터 높이).</summary>
        public void AttachCharacter(GameObject target, string name)
            => Build(target, name, _worldScale * _characterScaleMultiplier, _heightOffset);

        /// <summary>
        /// 대상 위에 이름표를 생성한다. name 이 비어 있거나 target 이 null 이면 아무것도 안 한다.
        /// (이미 이름표가 있어도 중복 생성하므로, 같은 대상에 한 번만 호출할 것)
        /// </summary>
        private void Build(GameObject target, string name, float worldScale, float heightOffset)
        {
            if (target == null || string.IsNullOrWhiteSpace(name)) return;

            // RectTransform + Canvas 를 보장해 World Space 캔버스 루트 생성
            var go = new GameObject($"NameTag_{name}", typeof(RectTransform), typeof(Canvas));
            var rt = go.GetComponent<RectTransform>();
            rt.sizeDelta   = _size;
            rt.localScale  = Vector3.one * worldScale;

            var canvas = go.GetComponent<Canvas>();
            canvas.renderMode  = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            // GraphicRaycaster 미부착 — 표시 전용(클릭 불필요) → EventSystem 레이캐스트 비용·NaN 회피

            if (_background)
            {
                var bgGo = new GameObject("BG", typeof(RectTransform));
                bgGo.transform.SetParent(go.transform, false);
                var img = bgGo.AddComponent<Image>();
                img.color         = _backgroundColor;
                img.raycastTarget = false;
                StretchFull(img.rectTransform);
            }

            var textGo = new GameObject("Text", typeof(RectTransform));
            textGo.transform.SetParent(go.transform, false);
            var tmp = textGo.AddComponent<TextMeshProUGUI>();
            if (_font != null) tmp.font = _font;
            tmp.text                = name;
            tmp.fontSize            = _fontSize;
            tmp.color               = _textColor;
            tmp.alignment           = TextAlignmentOptions.Center;
            tmp.enableWordWrapping  = false;
            tmp.raycastTarget       = false;
            StretchFull(tmp.rectTransform);

            var tag = go.AddComponent<NameTag>();
            tag.Bind(target.transform, heightOffset);
        }

        private static void StretchFull(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }
    }
}
