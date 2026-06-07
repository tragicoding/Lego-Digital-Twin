using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using LegoTwin.Character;
using LegoTwin.Managers;
using LegoTwin.UI;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// 자유 모드에서 현재 관람객이 자신의 창작물에 접근했을 때
    /// 동작 프롬프트 입력 · 시그니처 동작 설정 인터랙션을 제공한다.
    ///
    /// 이 컴포넌트는 PlazaManager.AttachViewToCurrentSession() 에서 런타임으로만 추가된다.
    /// 씬에 직접 배치하지 않는다.
    ///
    /// 동작 흐름:
    ///   1. 플레이어 접근  → interactPanel 표시
    ///   2. 버튼 클릭      → MotionPromptUI.Show()
    ///   3. 입력 확인      → PlacedCharacterController.PlayMotionFromPrompt()
    ///   4. confirmDelay 후 → SignatureMotionConfirmUI.Show()
    ///   5. 예 선택        → SessionManager.SetSignatureMotion() (Mock · Server 공통)
    ///
    /// SphereCollider 는 같은 GameObject 의 LikeSystem 이 보유하므로 별도 추가하지 않는다.
    /// </summary>
    public class OwnCreationInteractor : MonoBehaviour
    {
        [Tooltip("모션 재생 후 시그니처 확인 다이얼로그까지의 대기 시간 (초)")]
        public float confirmDelay = 3f;

        private GameObject                _interactPanel;
        private PlacedCharacterController _placedCharacter;
        private MotionPromptUI            _motionPromptUI;
        private SignatureMotionConfirmUI  _signatureConfirmUI;

        private bool _isInteracting;
        private bool _playerInRange;   // OnTriggerEnter 누락 방어용

        // ════════════════════════════════════════════════════════════
        // 초기화 — PlazaManager 에서 AddComponent 직후 호출
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// panel 이 null 이면 패널 없이 동작한다.
        /// confirmUI 가 null 이면 입력 즉시 저장한다.
        /// </summary>
        public void Setup(
            PlacedCharacterController character,
            GameObject panel,
            MotionPromptUI promptUI,
            SignatureMotionConfirmUI confirmUI)
        {
            _placedCharacter    = character;
            _motionPromptUI     = promptUI;
            _signatureConfirmUI = confirmUI;
            _interactPanel      = panel;

            if (_interactPanel == null) return;

            _interactPanel.SetActive(false);

            // 패널 내 버튼에 런타임으로 리스너 연결 (프리팹 OnClick 사전 설정 불필요)
            var btn = _interactPanel.GetComponentInChildren<Button>(includeInactive: true);
            btn?.onClick.AddListener(OnMotionInputPressed);
        }

        private void OnDestroy()
        {
            if (_interactPanel == null) return;
            var btn = _interactPanel.GetComponentInChildren<Button>(includeInactive: true);
            btn?.onClick.RemoveListener(OnMotionInputPressed);
        }

        // ════════════════════════════════════════════════════════════
        // 근접 감지 — 같은 GameObject 의 SphereCollider(LikeSystem) 가 트리거 발생
        // ════════════════════════════════════════════════════════════

        private void OnTriggerEnter(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            OnPlayerEnter();
        }

        // 가이드 모드 종료 시점에 플레이어가 이미 트리거 안에 있으면
        // OnTriggerEnter 가 발생하지 않을 수 있으므로 OnTriggerStay 로 보완한다.
        private void OnTriggerStay(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            if (!_playerInRange) OnPlayerEnter();
        }

        private void OnTriggerExit(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            _playerInRange = false;
            if (_interactPanel != null) _interactPanel.SetActive(false);
        }

        private void OnPlayerEnter()
        {
            _playerInRange = true;

            if (_interactPanel != null)
                _interactPanel.SetActive(true);     // 패널 모드: 버튼 UI 표시
            else
                OnMotionInputPressed();             // 폴백: 패널 없으면 입력창 직접 열기
        }

        // ════════════════════════════════════════════════════════════
        // 동작 입력 흐름
        // ════════════════════════════════════════════════════════════

        // interactPanel 의 "동작 입력하기" 버튼 → Setup() 에서 자동 연결됨
        private void OnMotionInputPressed()
        {
            if (_isInteracting) return;

            if (_motionPromptUI == null)
            {
                Debug.LogWarning("[OwnCreationInteractor] MotionPromptUI 미설정 " +
                                 "— PlazaManager.motionPromptUI 를 Inspector 에서 연결하세요.");
                return;
            }

            _isInteracting = true;
            _motionPromptUI.Show(OnMotionSubmitted);
        }

        private void OnMotionSubmitted(string input)
        {
            _placedCharacter?.PlayMotionFromPrompt(input);

            if (_signatureConfirmUI == null)
            {
                // 확인 다이얼로그 없으면 입력 즉시 저장
                SessionManager.Instance?.SetSignatureMotion(MotionPromptParser.Parse(input).ToString());
                _isInteracting = false;
                return;
            }

            StartCoroutine(ShowConfirmAfterDelay(input));
        }

        private IEnumerator ShowConfirmAfterDelay(string motionInput)
        {
            yield return new WaitForSeconds(confirmDelay);
            ShowSignatureConfirm(motionInput);
        }

        // 시그니처 설정 확인 다이얼로그 (예 / 아니오 / 나가기)
        private void ShowSignatureConfirm(string motionInput)
        {
            _signatureConfirmUI.Show(
                confirmed =>
                {
                    if (confirmed)
                    {
                        // 예: 시그니처 저장 후 종료
                        SessionManager.Instance?.SetSignatureMotion(
                            MotionPromptParser.Parse(motionInput).ToString());
                        _isInteracting = false;
                    }
                    else
                    {
                        // 아니오: 모션 프롬프트 재표시
                        _motionPromptUI?.Show(OnMotionSubmitted);
                    }
                },
                showExitButton: true,
                onExit: () => ShowExitConfirm(motionInput)   // 나가기: 종료 확인으로
            );
        }

        // 나가기 누를 때 표시하는 종료 확인 다이얼로그 (예 / 아니오)
        private void ShowExitConfirm(string motionInput)
        {
            _signatureConfirmUI.Show(
                confirmed =>
                {
                    if (confirmed)
                        _isInteracting = false;             // 예: 저장 없이 종료
                    else
                        ShowSignatureConfirm(motionInput);  // 아니오: 시그니처 확인으로 복귀
                },
                showExitButton: false,
                onExit: null,
                message: "입력 종료 시 이전 동작으로\n유지됩니다.",
                noCaption: "시그니처 동작 설정으로\n돌아갈래요"
            );
        }
    }
}
