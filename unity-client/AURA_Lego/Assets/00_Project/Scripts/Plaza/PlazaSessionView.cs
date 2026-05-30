using UnityEngine;
using TMPro;
using LegoTwin.Data;

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

        private LikeSystem _likeSystem;

        // ════════════════════════════════════════════════════════════
        // 초기화
        // ════════════════════════════════════════════════════════════

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
            if (starObject != null)
                starObject.SetActive(isTop);
        }
    }
}
