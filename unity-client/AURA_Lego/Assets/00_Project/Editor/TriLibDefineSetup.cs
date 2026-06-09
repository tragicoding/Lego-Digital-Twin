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
        private const string TriLibDefine = "TRILIB";
        private const string DisableEditorGltfImportDefine = "TRILIB_DISABLE_EDITOR_GLTF_IMPORT";

        static TriLibDefineSetup()
        {
            SyncDefines(TriLibPresent());
        }

        // 어셈블리 로드 순서·이름 변형에 견고하도록 타입 조회 + 어셈블리 스캔 병행.
        private static bool TriLibPresent()
        {
            if (Type.GetType("TriLibCore.AssetLoader, TriLibCore") != null) return true;
            return AppDomain.CurrentDomain.GetAssemblies()
                .Any(a => a.GetName().Name.StartsWith("TriLibCore", StringComparison.Ordinal));
        }

        private static void SyncDefines(bool triLibPresent)
        {
            var namedTarget = NamedBuildTarget.FromBuildTargetGroup(
                BuildPipeline.GetBuildTargetGroup(EditorUserBuildSettings.activeBuildTarget));

            var symbols = PlayerSettings.GetScriptingDefineSymbols(namedTarget);
            var list = symbols.Split(';').Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
            bool changed = false;

            changed |= SetSymbol(list, TriLibDefine, triLibPresent);

            // 역할 분담 고정:
            //   - 에디터/런타임 GLB(.gltf/.glb) = glTFast
            //   - 런타임 FBX                = TriLib
            // 따라서 TriLib 설치 환경에서는 TriLib의 에디터 glTF importer를 항상 끈다.
            changed |= SetSymbol(list, DisableEditorGltfImportDefine, triLibPresent);

            if (!changed) return;

            PlayerSettings.SetScriptingDefineSymbols(namedTarget, string.Join(";", list));
            UnityEngine.Debug.Log(
                $"[TriLibDefineSetup] define 동기화 완료 " +
                $"(target={namedTarget}, TriLib={triLibPresent}, " +
                $"{TriLibDefine}={list.Contains(TriLibDefine)}, " +
                $"{DisableEditorGltfImportDefine}={list.Contains(DisableEditorGltfImportDefine)})");
        }

        private static bool SetSymbol(System.Collections.Generic.List<string> list, string symbol, bool enable)
        {
            bool has = list.Contains(symbol);
            if (enable == has) return false;

            if (enable) list.Add(symbol);
            else        list.Remove(symbol);
            return true;
        }
    }
}
#endif
