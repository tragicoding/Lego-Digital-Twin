using System.Collections;
using UnityEngine;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// 좋아요 클릭 시 하트 오브젝트를 주변에 생성·애니메이션 후 자동 제거.
    ///
    /// 사용법:
    ///   LikeSystem과 같은 GameObject에 추가 → LikeSystem.Start()에서 자동 연결.
    ///   heartPrefab: World Space Canvas + TMP_Text("♥") 구성 프리팹.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] heartPrefab — 하트 프리팹 연결
    ///       구조: HeartParticle (Canvas - World Space, Scale 0.01)
    ///               └── HeartText (TextMeshProUGUI — "♥", 빨간색, FontSize 80)
    ///   [ ] spawnRadius / floatHeight — 씬 스케일에 맞게 조정
    /// </summary>
    public class HeartEffect : MonoBehaviour
    {
        [Header("하트 프리팹")]
        [Tooltip("World Space Canvas + TMP_Text '♥' 구성 프리팹")]
        [SerializeField] private GameObject heartPrefab;

        [Header("연출 설정")]
        [Tooltip("한 번에 생성할 하트 수")]
        [SerializeField] private int   heartCount  = 15;

        [Tooltip("스폰 랜덤 반경 (씬 스케일에 맞게 조정)")]
        [SerializeField] private float spawnRadius = 40f;

        [Tooltip("하트가 올라가는 높이")]
        [SerializeField] private float floatHeight = 100f;

        [Tooltip("애니메이션 지속 시간 (초)")]
        [SerializeField] private float duration    = 1.2f;

        [Tooltip("하트 간 스폰 간격 (초) — 0이면 동시 생성")]
        [SerializeField] private float spawnDelay   = 0.08f;

        [Tooltip("하트 크기 (프리팹 스케일 무시하고 이 값으로 강제 적용)")]
        [SerializeField] private float heartScale   = 0.1f;

        [Header("루프 설정 (StarObject용)")]
        [Tooltip("루프 모드 시 스폰 사이클 간격 (초)")]
        [SerializeField] private float loopInterval = 3.0f;

        private Coroutine _loopCoroutine;

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>자신의 위치에서 하트를 지속적으로 방출한다. StarObject에 붙여서 사용.</summary>
        public void PlayLoop()
        {
            if (heartPrefab == null) return;
            StopLoop();
            _loopCoroutine = StartCoroutine(LoopRoutine());
        }

        /// <summary>루프 방출을 중단한다.</summary>
        public void StopLoop()
        {
            if (_loopCoroutine != null) { StopCoroutine(_loopCoroutine); _loopCoroutine = null; }
        }

        private IEnumerator LoopRoutine()
        {
            while (true)
            {
                StartCoroutine(SpawnRoutine(transform.position));
                yield return new WaitForSeconds(loopInterval);
            }
        }

        /// <summary>
        /// LikeSystem.OnLikePressed()에서 호출. 하트 이펙트 시작.
        /// spawnPosition: 하트가 나올 월드 위치 (버튼 위치).
        /// </summary>
        public void Play(Vector3 spawnPosition)
        {
            if (heartPrefab == null)
            {
                Debug.LogWarning("[HeartEffect] heartPrefab 미연결 — Inspector에서 연결하세요.");
                return;
            }
            Debug.Log($"[HeartEffect] Play 호출 — spawnPos={spawnPosition}, heartCount={heartCount}");
            StartCoroutine(SpawnRoutine(spawnPosition));
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private IEnumerator SpawnRoutine(Vector3 spawnPosition)
        {
            for (int i = 0; i < heartCount; i++)
            {
                StartCoroutine(AnimateHeart(spawnPosition));
                if (spawnDelay > 0f)
                    yield return new WaitForSeconds(spawnDelay);
            }
        }

        private IEnumerator AnimateHeart(Vector3 spawnPosition)
        {
            // 버튼 위치 기준 랜덤 오프셋으로 스폰
            var offset = new Vector3(
                Random.Range(-spawnRadius, spawnRadius),
                Random.Range(0f,          spawnRadius * 0.5f),
                Random.Range(-spawnRadius, spawnRadius));

            var go = Instantiate(heartPrefab, spawnPosition + offset, Quaternion.identity);
            go.transform.localScale = Vector3.one * heartScale;

            // Billboard: 카메라 방향으로 정렬
            if (Camera.main != null)
                go.transform.forward = Camera.main.transform.forward;

            var startPos   = go.transform.position;
            var endPos     = startPos + Vector3.up * floatHeight;
            var startScale = go.transform.localScale;

            // CanvasGroup으로 페이드 아웃
            var cg = go.GetComponentInChildren<CanvasGroup>();
            if (cg == null)
            {
                var child = go.transform.childCount > 0 ? go.transform.GetChild(0).gameObject : go;
                cg = child.AddComponent<CanvasGroup>();
            }

            float elapsed = 0f;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / duration;

                go.transform.position   = Vector3.Lerp(startPos, endPos, t);
                // 처음엔 커졌다가 사라지는 스케일
                go.transform.localScale = startScale * Mathf.Lerp(1.2f, 0.6f, t);
                cg.alpha                = 1f - t;

                yield return null;
            }

            Destroy(go);
        }
    }
}
