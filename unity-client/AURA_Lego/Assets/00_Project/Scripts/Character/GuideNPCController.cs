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
    ///   - 완료 후 OnFreeModeSwitched → OnGuideFinished 순서로 이벤트 발행
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Inspector에 guideArrivalPoint, myCreationWaypoint Transform 연결
    ///   [ ] OnDialogueChanged 이벤트 구독해서 말풍선 UI 표시
    ///   [ ] OnFreeModeSwitched 이벤트 구독해서 팝업 표시
    ///   [ ] OnGuideFinished 이벤트 구독해서 자유 모드 씬/상태 전환
    ///   [ ] 캐릭터 Animator에 "walk", "idle" 트리거 등록
    /// </summary>
    public class GuideNPCController : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("Components")]
        public CharacterAnimationController Animation;

        [Header("이동 설정")]
        public float moveSpeed        = 5f;
        public float rotationSpeed    = 5f;
        public float arrivalThreshold = 0.15f;

        [Header("등장 연출")]
        [Tooltip("스폰 후 걸어가서 멈출 위치. 비워두면 스폰 위치에서 바로 인사.")]
        public Transform guideArrivalPoint;

        [Header("시나리오 웨이포인트")]
        [Tooltip("내 창작물(캐릭터+오브제) 앞 위치 — 순간이동 대상")]
        public Transform myCreationWaypoint;

        [Header("NPC 정보 (런타임 주입)")]
        [SerializeField] private string _npcName;
        [SerializeField] private string _sessionId;

        [Header("배치 캐릭터 (모션 재생 대상)")]
        [Tooltip("광장 오브제 옆 배치 캐릭터의 PlacedCharacterController 연결")]
        public PlacedCharacterController placedCharacter;


        [Header("플레이어 순간이동 위치")]
        [Tooltip("창작물 확인 시 플레이어가 이동할 위치. 비워두면 NPC와 동일 위치로 이동.")]
        public Transform playerTeleportPoint;

        // ── 이벤트 (UI 개발자가 구독) ────────────────────────────────

        /// <summary>말풍선 텍스트 변경 시 발생. UI에서 구독해 말풍선을 표시.</summary>
        public event Action<string> OnDialogueChanged;

        /// <summary>자유 모드 전환 직전 발생 — 팝업("자유모드로 전환되었습니다.") 표시용.</summary>
        public event Action OnFreeModeSwitched;

        /// <summary>가이드 시나리오 완료 → 자유 모드 전환 트리거.</summary>
        public event Action OnGuideFinished;

        /// <summary>순간이동 발생 시 발행. GameFlowManager에서 플레이어를 해당 위치로 이동시킨다.</summary>
        public event Action<Vector3> OnPlayerTeleportRequested;

        /// <summary>
        /// 모션 프롬프트 입력 요청 시 발생.
        /// 구독 예:
        ///   npc.OnMotionPromptRequested += callback => MyInputUI.Open(callback);
        /// 콜백으로 입력 문자열을 전달하면 PlacedCharacterController가 모션을 재생한다.
        /// </summary>
        public event Action<Action<string>> OnMotionPromptRequested;

        /// <summary>
        /// 모션 확인 후 시그니처 동작으로 설정할지 묻는 다이얼로그 요청.
        /// (motionInput: 입력된 프롬프트 텍스트, onResult: 예→true / 아니오→false 선택 후 호출)
        /// </summary>
        public event Action<string, Action<bool>> OnSignatureMotionConfirmRequested;

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
            // ── 0. 등장 — guideSpawnPoint 에서 guideArrivalPoint 까지 이동 후 플레이어 방향으로 회전
            if (guideArrivalPoint != null)
            {
                MoveTo(guideArrivalPoint.position);
                yield return WaitUntilArrived();
            }
            yield return FacePlayerRoutine();
            yield return new WaitForSeconds(0.3f);

            // ── 1. 입장 인사 ─────────────────────────────────────────
            Say("MINIVERSE에 오신 걸 환영해요!");
            yield return new WaitForSeconds(3f);

            Say($"저는 당신이 만든 \n'{_npcName}'(이)에요.");
            yield return new WaitForSeconds(3f);

            Say("광장에 가서 \n오브제들을 확인해볼까요?");
            yield return new WaitForSeconds(3f);

            // ── 2. 창작물 위치로 순간이동 ────────────────────────────
            if (myCreationWaypoint != null)
                TeleportTo(myCreationWaypoint.position);
            yield return FacePlayerRoutine();

            Say("짜잔! \n당신이 만든 오브제에요");
            yield return new WaitForSeconds(3f);

            // ── 3. 모션 프롬프트 입력 + 시그니처 설정 (아니오 선택 시 재시도) ──────────
            Say("캐릭터에는 \n원하는 동작을 입력할 수 있어요, \n한번 해볼까요?");
            yield return new WaitForSeconds(3f);

            bool signatureConfirmed = false;
            bool isRetry            = false;

            while (!signatureConfirmed)
            {
                if (isRetry)
                {
                    Say("다른 동작으로 \n다시 해볼까요?");
                    yield return new WaitForSeconds(2f);
                }

                // 모션 입력 대기
                string motionInput = null;
                OnMotionPromptRequested?.Invoke(text => motionInput = text);
                yield return new WaitUntil(() => motionInput != null);

                // 가이드 NPC는 idle 고정 — 배치 캐릭터(광장 FBX)만 모션 재생
                Animation?.animation_idle();
                if (placedCharacter != null)
                    placedCharacter.PlayMotionFromPrompt(motionInput);
                else
                    Debug.LogWarning("[GuideNPCController] placedCharacter가 연결되지 않았습니다.");

                yield return new WaitForSeconds(3f);

                if (!isRetry)
                {
                    Say("지금 이 동작을 \n 시그니처 동작으로 \n 설정해보세요!");
                    yield return new WaitForSeconds(3f);
                    Say("자유모드에서는 다른 사람들이 \n당신의 시그니처 동작을\n 볼 수 있답니다!");
                    yield return new WaitForSeconds(3f);
                }

                // 시그니처 확인 (안내 모드: 나가기 버튼 없음)
                if (OnSignatureMotionConfirmRequested != null)
                {
                    bool? confirmed = null;
                    OnSignatureMotionConfirmRequested.Invoke(motionInput, result => confirmed = result);
                    yield return new WaitUntil(() => confirmed.HasValue);
                    signatureConfirmed = confirmed.Value;
                }
                else
                {
                    signatureConfirmed = true;  // 구독자 없으면 스킵
                }

                isRetry = true;
            }

            // ── 4. 투표 설명 ─────────────────────────────────────────
            Say("광장에서는 \n다른 창작자들이 만든 오브제들도 \n볼 수 있어요.");
            yield return new WaitForSeconds(3f);

            Say("다가가서 마음에 드는 오브제에게 \n투표를 해보세요!");
            yield return new WaitForSeconds(3f);

            // ── 5. 자유 모드 전환 ────────────────────────────────────
            Debug.Log("[GuideNPCController] 시나리오 완료");
            OnFreeModeSwitched?.Invoke();
            yield return new WaitForSeconds(1.5f);
            OnGuideFinished?.Invoke();
        }

        // ════════════════════════════════════════════════════════════
        // 내부 이동 구현
        // ════════════════════════════════════════════════════════════

        /// <summary>NPC를 즉시 이동하고 플레이어 순간이동 이벤트를 발행한다.</summary>
        private void TeleportTo(Vector3 npcPosition)
        {
            StopMoving();
            transform.position = npcPosition;

            var playerDest = playerTeleportPoint != null
                ? playerTeleportPoint.position
                : npcPosition;
            OnPlayerTeleportRequested?.Invoke(playerDest);
        }

        /// <summary>Camera.main 방향으로 부드럽게 회전. duration 초 동안 Slerp.</summary>
        private IEnumerator FacePlayerRoutine(float duration = 0.4f)
        {
            var cam = Camera.main;
            if (cam == null) yield break;

            Quaternion startRot = transform.rotation;
            var dir = cam.transform.position - transform.position;
            dir.y = 0f;
            if (dir == Vector3.zero) yield break;

            Quaternion targetRot = Quaternion.LookRotation(dir);
            float elapsed = 0f;
            while (elapsed < duration)
            {
                transform.rotation = Quaternion.Slerp(startRot, targetRot, elapsed / duration);
                elapsed += Time.deltaTime;
                yield return null;
            }
            transform.rotation = targetRot;
        }

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
