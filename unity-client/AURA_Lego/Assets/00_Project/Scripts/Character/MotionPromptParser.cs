using System.Collections.Generic;
using UnityEngine;

namespace LegoTwin.Character
{
    /// <summary>
    /// 사용자 입력 문자열 → MotionType 변환 파서.
    ///
    /// 사용 예:
    ///   var type = MotionPromptParser.Parse("춤춰줘");  // → MotionType.Dance
    ///   var type = MotionPromptParser.Parse("알 수 없는 말"); // → MotionType.Idle
    ///
    /// 새 키워드 추가:
    ///   _keywordMap 딕셔너리에 항목 추가하면 됩니다.
    /// </summary>
    public static class MotionPromptParser
    {
        // 키워드(소문자) → MotionType 매핑
        // 앞에 위치한 키워드가 우선 매칭됩니다.
        private static readonly List<(string keyword, MotionType motion)> _keywordMap
            = new List<(string, MotionType)>
        {
            // ── 춤 ───────────────────────────────────────────────────
            ("춤춰",   MotionType.Dance),
            ("춤을",   MotionType.Dance),
            ("춤",     MotionType.Dance),
            ("댄스",   MotionType.Dance),
            ("춰줘",   MotionType.Dance),
            ("추자",   MotionType.Dance),
            ("춰",   MotionType.Dance),
            ("dance",  MotionType.Dance),

            // ── 달리기 ───────────────────────────────────────────────
            ("달려",   MotionType.Run),
            ("달리",   MotionType.Run),
            ("뛰어",   MotionType.Run),
            ("뛰기",   MotionType.Run),
            ("run",    MotionType.Run),

            // ── 걷기 / 산책 ──────────────────────────────────────────
            ("걸어",   MotionType.Walk),
            ("걸음",   MotionType.Walk),
            ("걷기",   MotionType.Walk),
            ("걷자",   MotionType.Walk),
            ("산책",   MotionType.Walk),
            ("walk",   MotionType.Walk),

            // ── 점프 ─────────────────────────────────────────────────
            ("점프",   MotionType.Jump),
            ("뛰어올", MotionType.Jump),
            ("jump",   MotionType.Jump),

            // ── 손흔들기 / 인사 ──────────────────────────────────────
            ("손흔",   MotionType.Wave),
            ("인사",   MotionType.Wave),
            ("하이",   MotionType.Wave),
            ("안녕",   MotionType.Wave),
            ("wave",   MotionType.Wave),

            // ── 앉기 ─────────────────────────────────────────────────
            ("앉아",   MotionType.Sit),
            ("앉기",   MotionType.Sit),
            ("sit",    MotionType.Sit),

            // ── 발차기 ───────────────────────────────────────────────
            ("발차기", MotionType.Kick),
            ("킥",     MotionType.Kick),
            ("kick",   MotionType.Kick),

            // ── 회전 ─────────────────────────────────────────────────
            ("돌아",   MotionType.Spin),
            ("돌기",   MotionType.Spin),
            ("회전",   MotionType.Spin),
            ("spin",   MotionType.Spin),

            // ── 환호 / 만세 ──────────────────────────────────────────
            ("만세",   MotionType.Cheer),
            ("환호",   MotionType.Cheer),
            ("cheer",  MotionType.Cheer),

            // ── 박수 ─────────────────────────────────────────────────
            ("박수",   MotionType.Clap),
            ("손뼉",   MotionType.Clap),
            ("clap",   MotionType.Clap),

            // ── 승리 ─────────────────────────────────────────────────
            ("승리",    MotionType.Victory),
            ("파이팅",  MotionType.Victory),
            ("화이팅",  MotionType.Victory),
            ("victory", MotionType.Victory),

            // ── 물구나무 / 핸드스탠드 ────────────────────────────────
            ("물구나무",   MotionType.Handstand),
            ("핸드스탠드", MotionType.Handstand),
            ("헤드스핀",   MotionType.Handstand),
            ("handstand",  MotionType.Handstand),
            ("headspin",   MotionType.Handstand),

            // ── 힙합 / 비보잉 ────────────────────────────────────────
            ("힙합",       MotionType.Hiphop),
            ("비보잉",     MotionType.Hiphop),
            ("hiphop",     MotionType.Hiphop),

            // ── 눕기 ─────────────────────────────────────────────────
            ("누워",       MotionType.Lie),
            ("눕",         MotionType.Lie),
            ("lie",        MotionType.Lie),

            // ── 울음 / 슬픔 ──────────────────────────────────────────
            ("울어",   MotionType.Cry),
            ("울음",   MotionType.Cry),
            ("울지",   MotionType.Cry),
            ("눈물",   MotionType.Cry),
            ("슬퍼",   MotionType.Cry),
            ("슬프",   MotionType.Cry),
            ("cry",    MotionType.Cry),
            ("sad",    MotionType.Cry),

            // ── 백덤블링 / 뒤공중돌기 ────────────────────────────────────
            ("백덤블링",     MotionType.BackTumbling),
            ("뒤공중돌기",   MotionType.BackTumbling),
            ("뒤재주넘기",   MotionType.BackTumbling),
            ("뒤공중제비",   MotionType.BackTumbling),
            ("backtumbling", MotionType.BackTumbling),
            ("back tumbling",MotionType.BackTumbling),
            ("backflip",     MotionType.BackTumbling),

            // ── 텀블링 / 공중돌기 ────────────────────────────────────────
            ("텀블링",   MotionType.Tumbling),
            ("덤블링",   MotionType.Tumbling),
            ("공중돌기", MotionType.Tumbling),
            ("재주넘기", MotionType.Tumbling),
            ("공중제비", MotionType.Tumbling),
            ("tumbling", MotionType.Tumbling),
            ("tumble",   MotionType.Tumbling),

            // ── 짧은 키워드(부분 문자열 주의) — 맨 아래에 두어 최저 우선순위 ──
            // "hi"는 "hiphop" 등 다른 키워드에 포함되므로, 다른 매칭이 모두 실패한 뒤에만 Wave로.
            ("hi",     MotionType.Wave),
        };

        /// <summary>
        /// 입력 문자열을 파싱해 MotionType을 반환한다.
        /// 매칭 키워드가 없으면 MotionType.Idle 반환.
        /// </summary>
        public static MotionType Parse(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return MotionType.Idle;

            string lower = input.ToLower().Trim();

            foreach (var (keyword, motion) in _keywordMap)
            {
                if (lower.Contains(keyword))
                    return motion;
            }

            return MotionType.Idle;
        }
    }
}
