using System;
using System.Collections;
using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 가이드 NPC — 이동 / 말풍선 / 전체 시나리오 진행.
    ///
    /// 역할:
    ///   - 관람객이 만든 캐릭터가 월드를 안내하는 NPC
    ///   - 가이드 모드 시나리오를 순서대로 실행
    ///   - 완료 후 OnGuideFinished 이벤트 발생 → 자유 모드 전환
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Inspector에 Waypoint Transform 연결 (plazaPathWaypoints, myCreationWaypoint)
    ///   [ ] OnDialogueChanged 이벤트 구독해서 말풍선 UI 표시
    ///   [ ] OnBubbleTextInputRequested 이벤트 구독해서 입력창 UI 열기
    ///   [ ] OnGuideFinished 이벤트 구독해서 자유 모드 씬/상태 전환
    ///   [ ] 캐릭터 Animator에 "walk", "idle" 트리거 등록
    /// </summary>
    public class GuideNPCController : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("Components")]
        public CharacterAnimationController Animation;

        [Header("이동 설정")]
        public float moveSpeed        = 2f;
        public float rotationSpeed    = 5f;
        public float arrivalThreshold = 0.15f;

        [Header("시나리오 웨이포인트")]
        [Tooltip("광장으로 이동하는 경로 (순서대로 연결)")]
        public Transform[] plazaPathWaypoints;
        [Tooltip("내 창작물(캐릭터+오브제) 앞 위치")]
        public Transform   myCreationWaypoint;

        [Header("NPC 정보 (런타임 주입)")]
        [SerializeField] private string _npcName;
        [SerializeField] private string _sessionId;

        [Header("배치 캐릭터 (모션 재생 대상)")]
        [Tooltip("광장 오브제 옆 배치 캐릭터의 PlacedCharacterController 연결")]
        public PlacedCharacterController placedCharacter;

        // ── 이벤트 (UI 개발자가 구독) ────────────────────────────────

        /// <summary>말풍선 텍스트 변경 시 발생. UI에서 구독해 말풍선을 표시.</summary>
        public event Action<string> OnDialogueChanged;

        /// <summary>가이드 시나리오 완료 → 자유 모드 전환 트리거.</summary>
        public event Action OnGuideFinished;

        /// <summary>
        /// 인사말 입력 요청 시 발생.
        /// 구독 예:
        ///   npc.OnBubbleTextInputRequested += callback => MyInputUI.Open(callback);
        /// </summary>
        public event Action<Action<string>> OnBubbleTextInputRequested;

        /// <summary>
        /// 모션 프롬프트 입력 요청 시 발생.
        /// 구독 예:
        ///   npc.OnMotionPromptRequested += callback => MyInputUI.Open(callback);
        /// 콜백으로 입력 문자열을 전달하면 PlacedCharacterController가 모션을 재생한다.
        /// </summary>
        public event Action<Action<string>> OnMotionPromptRequested;

        private Coroutine _moveCoroutine;

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>SessionData로 NPC를 초기화. SessionManager.OnSessionLoaded에서 호출.</summary>
        public void Initialize(SessionData session)
        {
            _npcName   = session.character_npc_name;
            _sessionId = session.session_id;
            Animation?.Initialize(session.assets?.character, session.bubble_text);
            Debug.Log($"[GuideNPCController] 초기화 완료 — {_npcName}");
        }

        /// <summary>목표 위치로 이동 (도착 후 idle 자동 전환).</summary>
        public void MoveTo(Vector3 target)
        {
            if (_moveCoroutine != null) StopCoroutine(_moveCoroutine);
            _moveCoroutine = StartCoroutine(MoveRoutine(target));
        }

        /// <summary>이동 완료까지 대기. yield return 으로 사용.</summary>
        public IEnumerator WaitUntilArrived() =>
            new WaitUntil(() => _moveCoroutine == null);

        /// <summary>말풍선 텍스트 설정 + 로그 출력.</summary>
        public void Say(string text)
        {
            Debug.Log($"[{_npcName ?? "NPC"}] {text}");
            OnDialogueChanged?.Invoke(text);
        }

        /// <summary>이동 즉시 중단.</summary>
        public void StopMoving()
        {
            if (_moveCoroutine != null) { StopCoroutine(_moveCoroutine); _moveCoroutine = null; }
            Animation?.animation_idle();
        }

        // ════════════════════════════════════════════════════════════
        // 가이드 시나리오 진입점
        // SessionManager 또는 GameFlowManager에서 호출
        // ════════════════════════════════════════════════════════════

        public void StartGuideScenario() => StartCoroutine(GuideScenarioRoutine());

        // ════════════════════════════════════════════════════════════
        // 시나리오 본문 — 대사 타이밍/연출은 유니티 개발자가 조정
        // ════════════════════════════════════════════════════════════

        private IEnumerator GuideScenarioRoutine()
        {
            // ── 1. 입장 인사 ─────────────────────────────────────────
            yield return new WaitForSeconds(1f);
            Say($"안녕하세요, MINIVERSE에 온 걸 환영해요!");
            yield return new WaitForSeconds(3f);

            Say($"저는 당신이 만든 '{_npcName}'(이)에요. 만나서 반가워요!");
            yield return new WaitForSeconds(3f);

            Say("먼저 World를 소개할게요.");
            yield return new WaitForSeconds(3f);

            // ── 2. 월드 소개 ─────────────────────────────────────────
            // TODO: 유니티 개발자 — 월드 소개 대사 및 카메라 연출 추가
            Say("이곳은 4개의 구역으로 나눠져 있어요.");
            yield return new WaitForSeconds(3f);

            Say("해리포터, 디즈니, 미래도시, 동심 월드까지");
            yield return new WaitForSeconds(3f);

            Say("다양한 구역에는 즐길 수 있는 어트랙션도 있답니다.");
            yield return new WaitForSeconds(3f);

            // ── 3. 광장으로 이동 ─────────────────────────────────────
            Say("아참!  World를 즐기기 전에, 광장 구역으로 가볼까요?");
            yield return new WaitForSeconds(3f);
            
            Say("광장에서는 직접 만든 창작물들을 볼 수 있어요!");
            yield return new WaitForSeconds(3f);

            // TODO: 유니티 개발자 — plazaPathWaypoints 를 Inspector에 연결
            if (plazaPathWaypoints != null)
            {
                foreach (var wp in plazaPathWaypoints)
                {
                    if (wp == null) continue;
                    MoveTo(wp.position);
                    yield return WaitUntilArrived();
                }
            }

            Say("저기 당신이 만든 오브제가 보이네요, 가서 확인해볼까요?");
            yield return new WaitForSeconds(3f);

            // ── 4. 내 창작물로 이동 ──────────────────────────────────
            // TODO: 유니티 개발자 — myCreationWaypoint 를 Inspector에 연결
            if (myCreationWaypoint != null)
            {
                MoveTo(myCreationWaypoint.position);
                yield return WaitUntilArrived();
            }

            Say("여기가 바로 당신의 창작물이에요!");
            yield return new WaitForSeconds(3f);

            // ── 5. 모션 프롬프트 입력 ────────────────────────────────
            Say("여러분의 시그니처 동작을 캐릭터가 따라할거에요!");
            yield return new WaitForSeconds(3f);

            Say("원하는 동작을 입력해주세요! (예: 춤춰줘, 점프해, 손흔들어)");

            // UI에서 OnMotionPromptRequested 이벤트를 구독해 입력창을 열어야 합니다.
            // 구독 예:
            //   npc.OnMotionPromptRequested += callback => myVRInputUI.Open(text => callback(text));
            string motionInput = null;
            OnMotionPromptRequested?.Invoke(text => motionInput = text);
            yield return new WaitUntil(() => motionInput != null);

            // 배치 캐릭터에 모션 적용
            if (placedCharacter != null)
                placedCharacter.PlayMotionFromPrompt(motionInput);
            else
                Debug.LogWarning("[GuideNPCController] placedCharacter가 연결되지 않았습니다. Inspector에서 연결하세요.");

            yield return new WaitForSeconds(1f);

            Say("이후에 다른 사람들이 당신의 동작과 문구, 오브제를 보고 좋아요를 누를거에요.");
            yield return new WaitForSeconds(3f);

            Say("좋아요를 많이 받으면 전시회 끝난 후 소정의 상품이 있습니다!");
            yield return new WaitForSeconds(3f);

            // ── 6. 인사말 입력 (bubble_text) ─────────────────────────
            Say("다른 사람들에게 소개할 인사말을 입력해주세요!");

            // TODO: 유니티 개발자 — OnBubbleTextInputRequested 이벤트 구독해서 입력 UI 열기
            // npc.OnBubbleTextInputRequested += callback => MyInputUI.Open(text => callback(text));
            string inputtedText = null;
            OnBubbleTextInputRequested?.Invoke(text => inputtedText = text);
            yield return new WaitUntil(() => inputtedText != null);

            // 서버에 bubble_text 저장 (Server Mode에서만 실제 전송)
            var api = FindAnyObjectByType<Network.ApiClient>();
            if (api != null && !string.IsNullOrEmpty(_sessionId))
                yield return api.UpdateBubbleText(_sessionId, inputtedText);

            // ── 7. 자유 모드 전환 안내 ───────────────────────────────
            yield return new WaitForSeconds(0.5f);
            Say("자, 이제 자유롭게 월드를 탐험해보세요!");
            yield return new WaitForSeconds(2f);

            // ── 8. 종료 → 자유 모드 전환 ────────────────────────────
            Debug.Log("[GuideNPCController] 시나리오 완료");
            OnGuideFinished?.Invoke();
        }

        // ════════════════════════════════════════════════════════════
        // 내부 이동 구현
        // ════════════════════════════════════════════════════════════

        private IEnumerator MoveRoutine(Vector3 target)
        {
            Animation?.animation_walk();
            while (Vector3.Distance(transform.position, target) > arrivalThreshold)
            {
                transform.position = Vector3.MoveTowards(
                    transform.position, target, moveSpeed * Time.deltaTime);

                var dir = new Vector3(
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
    }
}
