using System.Collections.Generic;
using System.Text;

namespace LegoTwin.Core
{
    /// <summary>
    /// 두벌식 한글 입력 오토마타(조합기). 순수 C# — Unity 의존 없음(테스트·재사용 용이).
    ///
    /// 사용
    ///   Reset(현재 텍스트) 로 초기화 → 키 입력마다 Input(자모) / Append(영문·숫자·공백) / Backspace() 호출 →
    ///   Text 로 '확정 문자열 + 조합 중 음절'을 읽어 InputField 에 반영.
    ///
    /// 처리: 초성·중성·종성 조합, 겹받침(ㄳ ㄵ …), 복합 모음(ㅘ ㅝ …), 도깨비불(받침→다음 초성), 백스페이스 분해.
    /// 자모 char 단위로 입력받는다(키보드가 두벌식 자판의 자모를 보냄). 영문/숫자는 Append 로 그대로 누적.
    /// </summary>
    public class HangulComposer
    {
        private const string CHO  = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
        private const string JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ";
        private const string JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"; // index 0 = 받침 없음

        // 복합 모음: (기존 중성, 더한 중성) → 결과 중성
        private static readonly Dictionary<(int, int), int> VowelCombine = new();
        private static readonly Dictionary<int, int> VowelBase = new();        // 결과 중성 → 첫 모음(백스페이스 분해)
        // 겹받침: (기존 종성, 더한 자음의 종성) → 결과 종성
        private static readonly Dictionary<(int, int), int> FinalCombine = new();
        // 종성 분해: 결과 종성 → (남는 종성, 떨어져 나가 초성이 되는 자음의 초성 인덱스)
        private static readonly Dictionary<int, (int rem, int cho)> FinalSplit = new();

        static HangulComposer()
        {
            void V(char b, char a, char r) { int rb = JUNG.IndexOf(r); VowelCombine[(JUNG.IndexOf(b), JUNG.IndexOf(a))] = rb; VowelBase[rb] = JUNG.IndexOf(b); }
            V('ㅗ', 'ㅏ', 'ㅘ'); V('ㅗ', 'ㅐ', 'ㅙ'); V('ㅗ', 'ㅣ', 'ㅚ');
            V('ㅜ', 'ㅓ', 'ㅝ'); V('ㅜ', 'ㅔ', 'ㅞ'); V('ㅜ', 'ㅣ', 'ㅟ');
            V('ㅡ', 'ㅣ', 'ㅢ');

            void F(char baseFinal, char addCons, char result)
            {
                int rb = JONG.IndexOf(result);
                FinalCombine[(JONG.IndexOf(baseFinal), JONG.IndexOf(addCons))] = rb;
                FinalSplit[rb] = (JONG.IndexOf(baseFinal), CHO.IndexOf(addCons));
            }
            F('ㄱ', 'ㅅ', 'ㄳ'); F('ㄴ', 'ㅈ', 'ㄵ'); F('ㄴ', 'ㅎ', 'ㄶ');
            F('ㄹ', 'ㄱ', 'ㄺ'); F('ㄹ', 'ㅁ', 'ㄻ'); F('ㄹ', 'ㅂ', 'ㄼ'); F('ㄹ', 'ㅅ', 'ㄽ');
            F('ㄹ', 'ㅌ', 'ㄾ'); F('ㄹ', 'ㅍ', 'ㄿ'); F('ㄹ', 'ㅎ', 'ㅀ'); F('ㅂ', 'ㅅ', 'ㅄ');
        }

        private readonly StringBuilder _committed = new();
        private int _cho = -1, _jung = -1, _jong = 0;   // 조합 중 음절. _cho/_jung: -1=없음, _jong: 0=없음

        /// <summary>확정 문자열 + 조합 중 음절을 합친 현재 전체 텍스트.</summary>
        public string Text => _committed.ToString() + ComposingChar();

        /// <summary>주어진 텍스트로 초기화(모두 확정 상태, 조합 중 없음).</summary>
        public void Reset(string text)
        {
            _committed.Clear();
            if (!string.IsNullOrEmpty(text)) _committed.Append(text);
            _cho = -1; _jung = -1; _jong = 0;
        }

        /// <summary>두벌식 자모 한 글자 입력.</summary>
        public void Input(char jamo)
        {
            int v = JUNG.IndexOf(jamo);
            if (v >= 0) { InputVowel(v); return; }
            int c = CHO.IndexOf(jamo);
            if (c >= 0) { InputConsonant(jamo, c); return; }
            Append(jamo.ToString());   // 자모가 아니면 그냥 누적
        }

        /// <summary>영문·숫자·공백 등 비조합 문자를 그대로 누적(조합 중 음절은 먼저 확정).</summary>
        public void Append(string literal)
        {
            FlushComposing();
            _committed.Append(literal);
        }

        /// <summary>백스페이스: 조합 중이면 자모 단위로 분해, 아니면 확정 문자 1개 삭제.</summary>
        public void Backspace()
        {
            if (_jong > 0)
            {
                _jong = FinalSplit.TryGetValue(_jong, out var s) ? s.rem : 0;   // 겹받침→홑받침, 홑받침→없음
            }
            else if (_jung >= 0)
            {
                _jung = VowelBase.TryGetValue(_jung, out int b) ? b : -1;       // 복합모음→기본모음, 기본→없음
            }
            else if (_cho >= 0)
            {
                _cho = -1;
            }
            else if (_committed.Length > 0)
            {
                _committed.Remove(_committed.Length - 1, 1);
            }
        }

        // ── 내부 ─────────────────────────────────────────────────────

        private void InputConsonant(char jamo, int cho)
        {
            int cJong = JONG.IndexOf(jamo);   // 종성으로 쓸 수 있는 자음이면 >0 (ㄸㅃㅉ 등은 -1)

            if (_jung < 0)
            {
                // 중성 없음(빈 상태 or 초성만) → 초성 시작/교체
                if (_cho >= 0) FlushComposing();
                _cho = cho;
            }
            else if (_cho < 0)
            {
                // 홑모음만 있던 상태 → 확정하고 새 초성
                FlushComposing();
                _cho = cho;
            }
            else if (_jong == 0)
            {
                if (cJong > 0) _jong = cJong;          // 받침으로 결합
                else { FlushComposing(); _cho = cho; } // 받침 불가(ㄸㅃㅉ) → 확정 후 새 초성
            }
            else
            {
                // 이미 받침 있음 → 겹받침 시도, 안 되면 확정 후 새 초성
                if (cJong > 0 && FinalCombine.TryGetValue((_jong, cJong), out int merged)) _jong = merged;
                else { FlushComposing(); _cho = cho; }
            }
        }

        private void InputVowel(int v)
        {
            if (_jung < 0)
            {
                _jung = v;   // 초성만(또는 빈 상태)에 중성 부착 → 홑모음 or 초+중
            }
            else if (_jong == 0)
            {
                // 받침 없는 초+중(또는 홑모음) → 복합모음 시도, 안 되면 확정 후 새 홑모음
                if (VowelCombine.TryGetValue((_jung, v), out int merged)) _jung = merged;
                else { FlushComposing(); _jung = v; }
            }
            else
            {
                // 받침 있음 → 도깨비불: 받침(의 끝 자음)이 다음 음절 초성으로 이동
                var (rem, movedCho) = FinalSplit.TryGetValue(_jong, out var s)
                    ? s                                   // 겹받침: 끝 자음만 이동, 앞 자음은 남음
                    : (0, CHO.IndexOf(JONG[_jong]));      // 홑받침: 통째로 이동
                _jong = rem;
                FlushComposing();                          // 앞 음절 확정(_jong=rem 반영됨)
                _cho = movedCho; _jung = v;
            }
        }

        // 조합 중 음절을 확정 문자열에 붙이고 상태 초기화.
        private void FlushComposing()
        {
            string s = ComposingChar();
            if (s.Length > 0) _committed.Append(s);
            _cho = -1; _jung = -1; _jong = 0;
        }

        // 현재 조합 중 음절을 문자로(없으면 "").
        private string ComposingChar()
        {
            if (_cho >= 0 && _jung >= 0)
                return ((char)(0xAC00 + _cho * 588 + _jung * 28 + _jong)).ToString();
            if (_cho >= 0) return CHO[_cho].ToString();   // 초성만
            if (_jung >= 0) return JUNG[_jung].ToString(); // 홑모음
            return "";
        }
    }
}
