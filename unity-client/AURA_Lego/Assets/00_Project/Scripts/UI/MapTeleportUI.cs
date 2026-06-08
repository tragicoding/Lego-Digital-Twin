using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using LegoTwin.Core;

namespace LegoTwin.UI
{
    /// <summary>
    /// 지도 텔레포트 UI.
    /// 지도 버튼을 누르면 목적지 패널이 열리고, 미리 배치해 둔 각 목적지 버튼을 누르면
    /// 해당 위치로 순간이동한다.
    ///
    /// 설계 원칙:
    ///   - 버튼 자유 배치: 버튼을 런타임 생성하지 않는다. 씬에 직접 배치한 버튼을
    ///     목적지에 1:1로 연결만 한다 → Layout Group(Grid/Vertical 등) 없이
    ///     위치·크기·모양을 원하는 대로 배치할 수 있다.
    ///   - 의존성 최소화: 매니저 싱글톤 참조 없음. 플레이어 리그(Transform)와
    ///     목적지(Transform)만 안다. 실제 이동은 공용 PlayerTeleporter(CC-safe) 에
    ///     위임 → 텔레포트 로직 중복 없음.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] Player Rig 에 XR Origin(또는 Camera Rig 루트) 연결 (비우면 "Player" 태그로 자동 검색)
    ///   [ ] Map Button 에 지도 열기/닫기 토글 버튼 연결
    ///   [ ] Destination Panel 에 목적지 버튼들을 담는 패널(비활성으로 시작) 연결
    ///   [ ] Destinations 에 (직접 배치한 Button + 목표 위치 Transform) 항목 등록
    ///       · 버튼은 씬/패널 안에 자유롭게 배치 — Layout Group 불필요
    ///       · 버튼 텍스트·아이콘·위치·크기는 씬에서 직접 꾸민다
    ///       · 목표 위치: 씬에 빈 GameObject 를 배치해 연결
    ///
    /// Hierarchy 예시:
    ///   MapTeleportUI (Canvas)
    ///     ├── MapButton            (Button — 지도 열기/닫기)
    ///     └── DestinationPanel     (비활성으로 시작)
    ///          ├── Btn_광장        (Button — 자유 배치)
    ///          ├── Btn_입구        (Button — 자유 배치)
    ///          └── ...
    /// </summary>
    public class MapTeleportUI : MonoBehaviour
    {
        [Serializable]
        public struct Destination
        {
            [Tooltip("씬에 직접 배치한 목적지 버튼. 위치·크기·모양은 자유롭게 배치한다.")]
            public Button button;

            [Tooltip("이 버튼을 누르면 이동할 목표 위치. 씬에 빈 GameObject 를 배치해 연결한다.")]
            public Transform point;
        }

        [Header("플레이어")]
        [Tooltip("XR Origin(또는 Camera Rig 루트) Transform. 비우면 \"Player\" 태그 오브젝트의 루트로 자동 검색")]
        [SerializeField] private Transform _playerRig;

        [Header("UI 참조")]
        [Tooltip("지도 열기/닫기 토글 버튼")]
        [SerializeField] private Button _mapButton;

        [Tooltip("목적지 버튼들이 담기는 패널. 비활성으로 시작한다.")]
        [SerializeField] private GameObject _destinationPanel;

        [Header("목적지 (직접 배치한 버튼 ↔ 목표 위치)")]
        [Tooltip("씬에 배치한 버튼과 이동 목표를 1:1로 연결한다. 항목만 추가하면 자동 배선된다.")]
        [SerializeField] private Destination[] _destinations;

        [Header("지도 버튼 — 패널 열림 표시")]
        [Tooltip("DestinationPanel 이 열려 있을 때 MapButton 에 적용할 색. 닫히면 원래 색으로 복원된다.")]
        [SerializeField] private Color _openColor = new Color(0.45f, 0.7f, 1f, 1f);

        [Header("옵션")]
        [Tooltip("텔레포트 후 지도를 자동으로 닫는다")]
        [SerializeField] private bool _closeAfterTeleport = true;

        private bool _open;

        // MapButton 색 토글용 — Awake 에서 원래 ColorBlock 을 캡처해 열림/닫힘 두 벌을 만든다.
        private ColorBlock _closedColors;
        private ColorBlock _openColors;
        private bool       _mapButtonColorsReady;

        // 등록한 onClick 핸들러를 OnDestroy 에서 정확히 해제하기 위해 (버튼, 핸들러) 보관.
        // (인스펙터에서 사용자가 단 다른 onClick 은 건드리지 않는다 → RemoveAllListeners 미사용)
        private readonly List<(Button button, UnityAction handler)> _wired
            = new List<(Button, UnityAction)>();

        private void Awake()
        {
            CaptureMapButtonColors();
            if (_mapButton != null) _mapButton.onClick.AddListener(ToggleMap);
            WireDestinationButtons();
            SetOpen(false);   // 시작 시 닫힘 (원래 색)
        }

        private void OnDestroy()
        {
            if (_mapButton != null) _mapButton.onClick.RemoveListener(ToggleMap);

            foreach (var (button, handler) in _wired)
                if (button != null) button.onClick.RemoveListener(handler);
            _wired.Clear();
        }

        // ════════════════════════════════════════════════════════════
        // 배선 — 사전 배치된 버튼을 목적지에 1:1 연결 (생성·레이아웃 없음)
        // ════════════════════════════════════════════════════════════

        private void WireDestinationButtons()
        {
            foreach (var dest in _destinations)
            {
                if (dest.button == null)
                {
                    Debug.LogWarning("[MapTeleportUI] Destinations 에 버튼이 비어 있는 항목이 있습니다 — 건너뜀");
                    continue;
                }

                var captured = dest;   // 클로저 캡처 안전 (루프 변수 직접 캡처 회피)
                UnityAction handler = () => TeleportTo(captured);

                dest.button.onClick.AddListener(handler);
                _wired.Add((dest.button, handler));
            }
        }

        // ════════════════════════════════════════════════════════════
        // 토글 / 텔레포트
        // ════════════════════════════════════════════════════════════

        private void ToggleMap() => SetOpen(!_open);

        private void SetOpen(bool open)
        {
            _open = open;
            if (_destinationPanel != null) _destinationPanel.SetActive(open);
            ApplyMapButtonColor(open);
        }

        // ════════════════════════════════════════════════════════════
        // MapButton 색 — 패널 열림이면 _openColor, 닫힘이면 원래 색
        // ════════════════════════════════════════════════════════════

        // 원래 ColorBlock 을 캡처해 "열림용"(normal·selected 를 _openColor 로) 변형을 만들어 둔다.
        private void CaptureMapButtonColors()
        {
            if (_mapButton == null) return;

            _closedColors = _mapButton.colors;            // 인스펙터의 원래 색 그대로
            _openColors   = _closedColors;
            _openColors.normalColor   = _openColor;       // 평상시 색
            _openColors.selectedColor = _openColor;       // 클릭 후 선택 상태 색
            _mapButtonColorsReady = true;
        }

        private void ApplyMapButtonColor(bool open)
        {
            if (_mapButton == null || !_mapButtonColorsReady) return;

            _mapButton.colors = open ? _openColors : _closedColors;

            // colors 교체는 다음 상태 전이 때까지 그래픽에 반영되지 않으므로 즉시 한 번 직접 적용.
            if (_mapButton.targetGraphic != null)
                _mapButton.targetGraphic.color = open ? _openColor : _closedColors.normalColor;
        }

        private void TeleportTo(Destination dest)
        {
            if (dest.point == null)
            {
                Debug.LogWarning("[MapTeleportUI] 목적지의 목표 위치(point) 미연결 — 이동 생략");
                return;
            }

            var rig = ResolveRig();
            if (rig == null)
            {
                Debug.LogWarning("[MapTeleportUI] 플레이어 리그를 찾지 못함 — Player Rig 를 연결하세요.");
                return;
            }

            PlayerTeleporter.Teleport(rig, dest.point.position, dest.point.rotation);

            if (_closeAfterTeleport) SetOpen(false);
        }

        // _playerRig 미연결 시 "Player" 태그 오브젝트의 루트로 1회 자동 해석
        private Transform ResolveRig()
        {
            if (_playerRig != null) return _playerRig;

            var tagged = GameObject.FindGameObjectWithTag("Player");
            if (tagged != null) _playerRig = tagged.transform.root;
            return _playerRig;
        }
    }
}
