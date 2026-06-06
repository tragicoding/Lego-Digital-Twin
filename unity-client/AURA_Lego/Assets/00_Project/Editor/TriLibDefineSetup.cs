#if UNITY_EDITOR
using System;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;

namespace LegoTwin.EditorTools
{
    /// <summary>
    /// TriLib(유료 에셋, .gitignore 로 저장소에서 제외) 설치 여부에 따라
    /// TRILIB 스크립팅 define 을 자동으로 켜고 끈다.
    ///
    /// 목적
    ///   public 저장소에 define 을 하드코딩 커밋하지 않아도,
    ///     - TriLib 이 있는 환경 → TRILIB 정의 → TripoFbxLoader 실제 로드 경로 컴파일
    ///     - TriLib 이 없는 환경 → TRILIB 미정의 → 폴백 경로 컴파일(Mock)
    ///   각 머신이 자기 상태에 맞게 define 을 스스로 관리한다.
    ///
    /// 동작
    ///   에디터 로드·도메인 리로드(InitializeOnLoad)마다 TriLibCore 어셈블리
    ///   존재를 검사해 현재 빌드 타깃의 define 을 갱신한다. 플랫폼 전환 시에도
    ///   도메인 리로드로 재실행되어 자동 보정된다.
    /// </summary>
    [InitializeOnLoad]
    public static class TriLibDefineSetup
    {
        private const string Define = "TRILIB";

        static TriLibDefineSetup()
        {
            SetDefine(TriLibPresent());
        }

        // 어셈블리 로드 순서·이름 변형에 견고하도록 타입 조회 + 어셈블리 스캔 병행.
        private static bool TriLibPresent()
        {
            if (Type.GetType("TriLibCore.AssetLoader, TriLibCore") != null) return true;
            return AppDomain.CurrentDomain.GetAssemblies()
                .Any(a => a.GetName().Name.StartsWith("TriLibCore", StringComparison.Ordinal));
        }

        private static void SetDefine(bool enable)
        {
            var namedTarget = NamedBuildTarget.FromBuildTargetGroup(
                BuildPipeline.GetBuildTargetGroup(EditorUserBuildSettings.activeBuildTarget));

            var symbols = PlayerSettings.GetScriptingDefineSymbols(namedTarget);
            var list = symbols.Split(';').Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
            bool has = list.Contains(Define);

            if (enable == has) return;   // 이미 원하는 상태 → 변경 없음

            if (enable) list.Add(Define);
            else        list.Remove(Define);

            PlayerSettings.SetScriptingDefineSymbols(namedTarget, string.Join(";", list));
            UnityEngine.Debug.Log($"[TriLibDefineSetup] TRILIB define {(enable ? "추가됨" : "제거됨")} " +
                                  $"(target={namedTarget}). TriLib 설치 감지={enable}");
        }
    }
}
#endif
