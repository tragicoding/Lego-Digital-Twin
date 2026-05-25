using System.Collections;
using UnityEngine;
using LegoTwin.Network;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// 좋아요 시스템 — 관람객이 오브제에 접근 시 좋아요 가능.
    ///
    /// 동작:
    ///   - 관람객이 triggerRadius 이내로 접근 → 좋아요 UI 표시
    ///   - 관람객이 상호작용 → POST /sessions/{id}/like 호출
    ///   - 서버 응답으로 likes 수 업데이트 + PlazaSessionView 갱신
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] triggerRadius 값 조정 (기본 2m)
    ///   [ ] likeButtonUI 연결 (근접 시 표시할 하트/버튼 UI)
    ///   [ ] OnPlayerEnter / OnPlayerExit 로직 구현
    ///       (XR 컨트롤러 또는 플레이어 Collider 기반으로 트리거 감지)
    ///   [ ] 같은 방문자가 중복 좋아요 못 하도록 PlayerPrefs 또는 세션 기반 중복 방지 추가
    /// </summary>
    public class LikeSystem : MonoBehaviour
    {
        [Header("감지 설정")]
        [Tooltip("좋아요 가능 범위 (미터)")]
        public float triggerRadius = 2f;

        [Header("UI")]
        [Tooltip("근접 시 표시할 좋아요 버튼/하트 UI")]
        public GameObject likeButtonUI;

        private string _sessionId;
        private int    _likes;
        private bool   _playerNearby;

        // ════════════════════════════════════════════════════════════
        // 초기화 (PlazaSessionView에서 호출)
        // ════════════════════════════════════════════════════════════

        public void Initialize(string sessionId, int currentLikes)
        {
            _sessionId = sessionId;
            _likes     = currentLikes;
            if (likeButtonUI != null) likeButtonUI.SetActive(false);
        }

        public void UpdateCount(int likes) => _likes = likes;

        // ════════════════════════════════════════════════════════════
        // 근접 감지 (Collider Trigger 방식)
        // ════════════════════════════════════════════════════════════

        // TODO: 유니티 개발자 — XR 환경에 맞게 트리거 방식 선택
        // 옵션 A: Sphere Collider (Is Trigger) + OnTriggerEnter/Exit
        // 옵션 B: 매 프레임 Distance 체크 (VR 카메라 기준)

        private void OnTriggerEnter(Collider other)
        {
            // TODO: 유니티 개발자 — 플레이어 태그 확인 후 활성화
            if (!other.CompareTag("Player")) return;
            _playerNearby = true;
            if (likeButtonUI != null) likeButtonUI.SetActive(true);
        }

        private void OnTriggerExit(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            _playerNearby = false;
            if (likeButtonUI != null) likeButtonUI.SetActive(false);
        }

        // ════════════════════════════════════════════════════════════
        // 좋아요 실행
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 관람객이 좋아요를 누를 때 호출.
        /// 좋아요 버튼 UI의 OnClick 이벤트에 연결하거나 직접 호출.
        /// </summary>
        public void OnLikePressed()
        {
            if (string.IsNullOrEmpty(_sessionId)) return;
            StartCoroutine(SendLike());
        }

        private IEnumerator SendLike()
        {
            if (likeButtonUI != null) likeButtonUI.SetActive(false); // 중복 방지 임시 비활성

            yield return ApiClient.Instance.LikeSession(_sessionId, result =>
            {
                Debug.Log($"[LikeSystem] 좋아요 완료: {result.session_id} → {result.likes}");
                // WebSocket이 있으면 HandleMessage로 자동 갱신됨.
                // 없으면 PlazaSessionView 직접 갱신.
                var view = GetComponent<PlazaSessionView>();
                view?.UpdateLikes(result.likes);
                view?.SetTopLiked(result.is_top_liked);
            });
        }
    }
}
