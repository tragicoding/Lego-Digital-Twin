using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using LegoTwin.Managers;
using LegoTwin.Network;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// 좋아요 시스템 — 관람객이 오브제에 접근 시 좋아요 가능.
    ///
    /// 동작:
    ///   - 관람객이 triggerRadius 이내로 접근 → 좋아요 UI 표시
    ///   - 버튼 클릭 → 좋아요 처리 (세션당 1회 제한)
    ///   - 중복 시 alreadyLikedUI 안내 표시 후 자동 숨김
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] likeButtonUI    연결 (근접 시 표시할 하트/버튼 UI)
    ///   [ ] alreadyLikedUI 연결 ("이미 하트를 누른 오브제입니다" 안내 UI)
    /// </summary>
    public class LikeSystem : MonoBehaviour
    {
        [Header("감지 설정")]
        [Tooltip("좋아요 가능 범위 (미터)")]
        public float triggerRadius = 2f;

        [Header("UI")]
        [Tooltip("근접 시 표시할 좋아요 버튼/하트 UI")]
        public GameObject likeButtonUI;

        [Tooltip("중복 투표 시 표시할 안내 UI ('이미 하트를 누른 오브제입니다')")]
        public GameObject alreadyLikedUI;

        [Tooltip("중복 안내 표시 시간 (초)")]
        public float alreadyLikedDisplayDuration = 2f;

        // 이번 플레이 세션에서 좋아요를 누른 session_id 목록.
        // static이므로 모든 LikeSystem 인스턴스가 공유 (앱 재시작 시 자동 초기화).
        private static readonly HashSet<string> _likedSessionIds = new();

        private string        _sessionId;
        private int           _likes;
        private Camera        _cam;
        private RectTransform _likeButtonRect;

        // ════════════════════════════════════════════════════════════
        // 초기화 (PlazaSessionView에서 호출)
        // ════════════════════════════════════════════════════════════

        private void Start()
        {
            _cam = Camera.main;
            // likeButtonUI 안의 Button RectTransform을 캐시 (마우스 감지용)
            if (likeButtonUI != null)
            {
                var btn = likeButtonUI.GetComponentInChildren<Button>(includeInactive: true);
                if (btn != null) _likeButtonRect = btn.GetComponent<RectTransform>();
            }
        }

        public void Initialize(string sessionId, int currentLikes)
        {
            _sessionId = sessionId;
            _likes     = currentLikes;
            if (likeButtonUI    != null) likeButtonUI.SetActive(false);
            if (alreadyLikedUI  != null) alreadyLikedUI.SetActive(false);
        }

        public void UpdateCount(int likes) => _likes = likes;

        // ════════════════════════════════════════════════════════════
        // 마우스 클릭 감지 (EventSystem 미설정 환경 대응)
        // RectTransformUtility로 직접 버튼 영역을 감지해 OnLikePressed 호출
        // ════════════════════════════════════════════════════════════

        private void Update()
        {
            if (!Input.GetMouseButtonDown(0)) return;
            if (likeButtonUI == null || !likeButtonUI.activeSelf) return;
            if (_likeButtonRect == null) return;

            if (_cam == null) _cam = Camera.main;
            if (_cam == null) return;

            if (RectTransformUtility.RectangleContainsScreenPoint(
                    _likeButtonRect, Input.mousePosition, _cam))
            {
                OnLikePressed();
            }
        }

        // ════════════════════════════════════════════════════════════
        // 근접 감지
        // ════════════════════════════════════════════════════════════

        private void OnTriggerEnter(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            if (likeButtonUI != null) likeButtonUI.SetActive(true);
        }

        private void OnTriggerExit(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            if (likeButtonUI    != null) likeButtonUI.SetActive(false);
            if (alreadyLikedUI  != null) alreadyLikedUI.SetActive(false);
        }

        // ════════════════════════════════════════════════════════════
        // 좋아요 실행
        // ════════════════════════════════════════════════════════════

        /// <summary>좋아요 버튼 UI의 OnClick 이벤트에 연결하거나 직접 호출.</summary>
        public void OnLikePressed()
        {
            if (string.IsNullOrEmpty(_sessionId)) return;

            // 중복 투표 방지
            if (_likedSessionIds.Contains(_sessionId))
            {
                StartCoroutine(ShowAlreadyLikedMessage());
                return;
            }

            StartCoroutine(SendLike());
        }

        private IEnumerator SendLike()
        {
            if (likeButtonUI != null) likeButtonUI.SetActive(false);

            // ── Mock Mode ─────────────────────────────────────────────
            if (SessionManager.Instance?.dataSourceMode == DataSourceMode.Mock)
            {
                _likes++;
                _likedSessionIds.Add(_sessionId);
                var view = GetComponent<PlazaSessionView>();
                view?.UpdateLikes(_likes);
                PlazaManager.Instance?.HandleMockLike(_sessionId, _likes);
                Debug.Log($"[LikeSystem] Mock 좋아요: {_sessionId} → {_likes}");
                yield break;
            }

            // ── Server Mode ───────────────────────────────────────────
            yield return ApiClient.Instance.LikeSession(_sessionId, result =>
            {
                _likedSessionIds.Add(_sessionId);
                Debug.Log($"[LikeSystem] 좋아요 완료: {result.session_id} → {result.likes}, 1위: {result.top_session_id}");
                PlazaManager.Instance?.HandleServerLike(result);
            });
        }

        // 중복 투표 안내 표시 후 자동 숨김
        private IEnumerator ShowAlreadyLikedMessage()
        {
            if (alreadyLikedUI == null) yield break;
            alreadyLikedUI.SetActive(true);
            yield return new WaitForSeconds(alreadyLikedDisplayDuration);
            alreadyLikedUI.SetActive(false);
        }
    }
}
