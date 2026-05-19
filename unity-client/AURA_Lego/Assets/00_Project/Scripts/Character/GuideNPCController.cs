using System.Collections;
using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 가이드 NPC 이동 / 말풍선 / 시나리오 컨트롤러.
    /// 팀원이 씬에서 캐릭터 동선과 시나리오를 작성할 때 사용한다.
    ///
    /// 사용 예:
    ///   npc.SetNpcName("몽글이");
    ///   npc.SetBubbleText("안녕하세요!");
    ///   npc.MoveTo(targetTransform.position);
    ///   npc.StartGuideScenario();
    /// </summary>
    public class GuideNPCController : MonoBehaviour
    {
        [Header("Components")]
        public CharacterAnimationController Animation;

        [Header("Movement")]
        public float moveSpeed = 2f;
        public float rotationSpeed = 5f;
        public float arrivalThreshold = 0.15f;

        [Header("NPC Info (런타임 주입)")]
        [SerializeField] private string _npcName;
        [SerializeField] private string _bubbleText;

        private Coroutine _moveCoroutine;

        // ── 공개 API ────────────────────────────────────────────────

        public void SetNpcName(string name)
        {
            _npcName = name;
            if (Animation != null) Animation.npcName = name;
        }

        public void SetBubbleText(string text)
        {
            _bubbleText = text;
        }

        /// <summary>목표 위치로 이동. 이동 중에는 walk, 도착 후에는 idle.</summary>
        public void MoveTo(Vector3 targetPosition)
        {
            if (_moveCoroutine != null) StopCoroutine(_moveCoroutine);
            _moveCoroutine = StartCoroutine(MoveRoutine(targetPosition));
        }

        /// <summary>이동 즉시 중단 후 idle.</summary>
        public void StopMoving()
        {
            if (_moveCoroutine != null)
            {
                StopCoroutine(_moveCoroutine);
                _moveCoroutine = null;
            }
            Animation?.animation_idle();
        }

        /// <summary>
        /// 가이드 시나리오 진입점.
        /// 팀원이 이 함수 안에 캐릭터 동선 / 말풍선 타이밍을 작성한다.
        /// </summary>
        public void StartGuideScenario()
        {
            StartCoroutine(GuideScenarioRoutine());
        }

        // ── SessionData로 초기화 ────────────────────────────────────

        public void Initialize(CharacterAssetData data, string bubbleText = "")
        {
            SetNpcName(data.npc_name);
            SetBubbleText(bubbleText);
            Animation?.Initialize(data, bubbleText);
        }

        // ── 내부 구현 ────────────────────────────────────────────────

        private IEnumerator MoveRoutine(Vector3 target)
        {
            Animation?.animation_walk();

            while (Vector3.Distance(transform.position, target) > arrivalThreshold)
            {
                transform.position = Vector3.MoveTowards(
                    transform.position, target, moveSpeed * Time.deltaTime);

                Vector3 dir = new Vector3(
                    target.x - transform.position.x, 0f,
                    target.z - transform.position.z);

                if (dir != Vector3.zero)
                    transform.rotation = Quaternion.Slerp(
                        transform.rotation,
                        Quaternion.LookRotation(dir),
                        rotationSpeed * Time.deltaTime);

                yield return null;
            }

            _moveCoroutine = null;
            Animation?.animation_idle();
        }

        /// <summary>
        /// TODO: 팀원이 여기에 시나리오를 작성한다.
        /// 예시) 등장 → 오브제 위치로 이동 → 말풍선 출력 → idle
        /// </summary>
        private IEnumerator GuideScenarioRoutine()
        {
            Debug.Log($"[GuideNPCController] {_npcName} 시나리오 시작");

            // ── 시나리오 작성 위치 ──────────────────────────────────
            // yield return new WaitForSeconds(1f);
            // MoveTo(waypointA.position);
            // yield return new WaitUntil(() => _moveCoroutine == null);
            // SetBubbleText("안녕하세요!");
            // yield return new WaitForSeconds(2f);
            // ────────────────────────────────────────────────────────

            yield return null;
            Debug.Log($"[GuideNPCController] {_npcName} 시나리오 완료");
        }
    }
}
