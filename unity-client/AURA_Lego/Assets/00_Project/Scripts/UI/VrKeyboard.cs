using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using LegoTwin.Core;

namespace LegoTwin.UI
{
    /// <summary>
    /// VR 전용 World Space 한글/영문 키보드. 포커스된 TMP_InputField 를 자동 감지해 입력한다.
    ///
    /// 설계 (저결합·무중복·확장·서버동일)
    ///   - VR 모드에서만 동작(XRModeManager.Mode). 데스크톱은 물리 키보드 → 이 컴포넌트가 스스로 꺼짐(무회귀).
    ///   - UI 로직 스크립트(MotionPromptUI·WaitingScreenUI)는 전혀 참조하지 않는다. 어떤 TMP_InputField 든
    ///     포커스되면 자동으로 그 필드에 입력한다(FreeCameraController.IsTypingInInputField 와 동일 발상).
    ///   - 한글 조합은 HangulComposer(순수 로직)에 위임 — 키보드는 키 UI·라우팅만(단일 책임).
    ///   - 데이터 소스(Mock/Server) 미참조 → 모드(하드웨어)로만 결정 → Mock·Server 동일.
    ///   - 키는 레이아웃 문자열에서 자동 생성(키 프리팹 1개만 제공). 자판을 바꾸려면 인스펙터 문자열만 수정(확장).
    ///
    /// 레이아웃 토큰: "base" 또는 "base/shift" (공백 구분). 특수키 행(Shift·한/영·Space·⌫·완료)은 코드가 추가.
    /// 제출은 키보드가 아니라 각 UI 의 기존 버튼(다음/확인)으로 한다 — 그 버튼은 VR 레이로 눌림(ModeAdaptiveCanvas).
    ///
    /// 씬 구성: World Space Canvas 에 부착 + _root(본체)·_rowsParent(행 컨테이너)·_keyPrefab(라벨 TMP 자식 포함) 연결.
    /// </summary>
    public class VrKeyboard : MonoBehaviour
    {
        [Header("구성")]
        [Tooltip("키보드 본체(보이기/숨기기 대상). 비활성으로 시작.")]
        [SerializeField] private GameObject _root;
        [Tooltip("키 행들이 생성될 컨테이너")]
        [SerializeField] private RectTransform _rowsParent;
        [Tooltip("키 버튼 템플릿(자식에 TMP_Text 라벨 포함)")]
        [SerializeField] private Button _keyPrefab;
        [SerializeField] private float _rowHeight = 90f;
        [SerializeField] private float _keySpacing = 6f;

        [Header("VR 배치 (카메라 앞 추적)")]
        [SerializeField] private float _distance = 1.2f;
        [SerializeField] private Vector3 _localOffset = new Vector3(0f, -0.35f, 0f);
        [SerializeField] private float _followLerp = 8f;
        [SerializeField] private Camera _camera;

        [Header("자판 (base 또는 base/shift, 공백 구분)")]
        [SerializeField] private string[] _korRows =
        {
            "ㅂ/ㅃ ㅈ/ㅉ ㄷ/ㄸ ㄱ/ㄲ ㅅ/ㅆ ㅛ ㅕ ㅑ ㅐ/ㅒ ㅔ/ㅖ",
            "ㅁ ㄴ ㅇ ㄹ ㅎ ㅗ ㅓ ㅏ ㅣ",
            "ㅋ ㅌ ㅊ ㅍ ㅠ ㅜ ㅡ",
        };
        [SerializeField] private string[] _engRows =
        {
            "q/Q w/W e/E r/R t/T y/Y u/U i/I o/O p/P",
            "a/A s/S d/D f/F g/G h/H j/J k/K l/L",
            "z/Z x/X c/C v/V b/B n/N m/M",
        };

        private readonly HangulComposer _composer = new();
        private readonly List<(TMP_Text label, string baseV, string shiftV)> _charKeys = new();
        private TMP_InputField _field;
        private bool _active, _shift, _english;

        private void Awake()
        {
            if (_root != null) _root.SetActive(false);
        }

        private void Update()
        {
            if (!_active)
            {
                if (!XRModeManager.Resolved) return;
                if (XRModeManager.Mode != AppMode.VR) { enabled = false; return; }   // 데스크톱: 불필요
                Rebuild();
                _active = true;
            }

            TrackFocus();
            if (_root != null && _root.activeSelf) Follow();
        }

        // ── 포커스 추적 ───────────────────────────────────────────────
        private void TrackFocus()
        {
            var es = EventSystem.current;
            var go = es != null ? es.currentSelectedGameObject : null;
            var f = go != null ? go.GetComponent<TMP_InputField>() : null;

            // 새 입력창이 포커스되면 그 필드로 전환(키 버튼이 선택돼 f==null 이어도 현재 필드 유지).
            if (f != null && f != _field)
            {
                _field = f;
                _composer.Reset(_field.text);
                Show();
            }
            // 입력창이 닫히면(패널 비활성) 키보드 숨김.
            if (_field != null && !_field.isActiveAndEnabled)
            {
                _field = null;
                Hide();
            }
        }

        // ── 키 처리 ──────────────────────────────────────────────────
        private void OnCharKey(string value)
        {
            if (_field == null) return;

            if (_english) _composer.Append(value);     // 영문/숫자: 그대로 누적
            else if (value.Length == 1) _composer.Input(value[0]);  // 한글 자모
            else _composer.Append(value);

            if (_shift) SetShift(false);               // 시프트는 한 글자 후 해제
            ApplyToField();
        }

        private void OnSpace()     { if (_field == null) return; _composer.Append(" "); ApplyToField(); }
        private void OnBackspace() { if (_field == null) return; _composer.Backspace();  ApplyToField(); }

        private void OnDone()
        {
            Hide();
            if (EventSystem.current != null) EventSystem.current.SetSelectedGameObject(null);
            _field = null;
        }

        private void ApplyToField()
        {
            string t = _composer.Text;
            _field.text = t;
            _field.caretPosition = t.Length;   // 다시 포커스될 때 끝으로
        }

        // ── 시프트 / 한·영 ───────────────────────────────────────────
        private void SetShift(bool on)
        {
            _shift = on;
            foreach (var (label, baseV, shiftV) in _charKeys)
                label.text = _shift ? shiftV : baseV;
        }

        private void ToggleLanguage()
        {
            _english = !_english;
            _shift = false;
            _composer.Append("");   // 언어 전환 시 조합 중 음절 확정
            Rebuild();
        }

        // ── 표시 / 추적 ──────────────────────────────────────────────
        private void Show()
        {
            if (_root == null || _root.activeSelf) return;
            _root.SetActive(true);
            SnapInFront();
        }

        private void Hide()
        {
            if (_root != null) _root.SetActive(false);
        }

        private Camera Cam => _camera != null ? _camera : Camera.main;

        private void SnapInFront()
        {
            var cam = Cam;
            if (cam == null) return;
            transform.position = TargetPos(cam);
            transform.rotation = Quaternion.LookRotation(transform.position - cam.transform.position);
        }

        private void Follow()
        {
            var cam = Cam;
            if (cam == null) return;
            Vector3 target = TargetPos(cam);
            Quaternion look = Quaternion.LookRotation(target - cam.transform.position);
            float k = Time.unscaledDeltaTime * _followLerp;
            transform.position = Vector3.Lerp(transform.position, target, k);
            transform.rotation = Quaternion.Slerp(transform.rotation, look, k);
        }

        private Vector3 TargetPos(Camera cam)
        {
            var t = cam.transform;
            return t.position + t.forward * _distance + t.right * _localOffset.x + t.up * _localOffset.y;
        }

        // ── 키 생성 ──────────────────────────────────────────────────
        private void Rebuild()
        {
            if (_rowsParent == null || _keyPrefab == null) return;

            for (int i = _rowsParent.childCount - 1; i >= 0; i--)
                Destroy(_rowsParent.GetChild(i).gameObject);
            _charKeys.Clear();

            foreach (string row in (_english ? _engRows : _korRows))
            {
                var rowT = NewRow();
                foreach (string token in row.Split(' '))
                {
                    if (string.IsNullOrWhiteSpace(token)) continue;
                    string[] bs = token.Split('/');
                    string baseV = bs[0];
                    string shiftV = bs.Length > 1 ? bs[1] : bs[0];
                    var label = NewKey(rowT, _shift ? shiftV : baseV, () => OnCharKey(_shift ? shiftV : baseV));
                    _charKeys.Add((label, baseV, shiftV));
                }
            }

            // 특수키 행
            var special = NewRow();
            NewKey(special, "⇧",   () => SetShift(!_shift));
            NewKey(special, _english ? "한" : "A", ToggleLanguage);
            NewKey(special, "공백", OnSpace);
            NewKey(special, "⌫",   OnBackspace);
            NewKey(special, "완료", OnDone);
        }

        private RectTransform NewRow()
        {
            var go = new GameObject("Row", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            var rt = (RectTransform)go.transform;
            rt.SetParent(_rowsParent, false);
            var h = go.GetComponent<HorizontalLayoutGroup>();
            h.spacing = _keySpacing;
            h.childAlignment = TextAnchor.MiddleCenter;
            h.childForceExpandWidth = false; h.childForceExpandHeight = true;
            go.GetComponent<LayoutElement>().preferredHeight = _rowHeight;
            return rt;
        }

        private TMP_Text NewKey(RectTransform row, string label, Action onClick)
        {
            var btn = Instantiate(_keyPrefab, row);
            btn.gameObject.SetActive(true);
            btn.onClick.AddListener(() => onClick());
            var tmp = btn.GetComponentInChildren<TMP_Text>();
            if (tmp != null) tmp.text = label;
            return tmp;
        }
    }
}
