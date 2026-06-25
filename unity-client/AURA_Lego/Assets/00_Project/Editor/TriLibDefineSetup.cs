#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

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
    /// 왜 ProjectSettings 가 아니라 csc.rsp 인가 (재발 방지)
    ///   과거에는 PlayerSettings(=ProjectSettings.asset)에 TRILIB 를 써넣었는데,
    ///   ProjectSettings.asset 은 '추적되는' 파일이라 TriLib 설치 PC 에서 그 변경이
    ///   실수로 커밋되면 → TriLib 미설치 PC 가 받았을 때 #if TRILIB(using TriLibCore)
    ///   가 켜져 컴파일 에러 → Unity Safe Mode 에 빠졌다(2026-06-25 사고).
    ///   그래서 TRILIB 는 .gitignore 된 Assets/csc.rsp 에만 기록한다.
    ///   머신 로컬에만 존재 → 저장소로 절대 새지 않음 → 구조적으로 재발 불가.
    ///
    /// 적용 범위
    ///   Assets/csc.rsp 는 Assembly-CSharp(우리 런타임 코드) 컴파일에 -define 을 준다.
    ///   TripoFbxLoader 등 우리 코드가 전부 Assembly-CSharp 소속이라 정확히 커버된다.
    ///   에디터 Play 모드·플레이어 빌드 양쪽 컴파일에 모두 반영된다.
    ///   ※ TRILIB_DISABLE_EDITOR_GLTF_IMPORT 는 TriLib '자체' 에디터 asmdef 가
    ///     소비하므로 csc.rsp 로는 닿지 않는다. 그 define 은 전역(ProjectSettings)에
    ///     영구 보존한다(미설치 PC 에선 소비처가 없어 무해 → Safe Mode 유발 안 함).
    /// </summary>
    [InitializeOnLoad]
    public static class TriLibDefineSetup
    {
        private const string TriLibDefine = "TRILIB";
        private const string RspLine = "-define:" + TriLibDefine;
        private static string RspFullPath => Path.Combine(UnityEngine.Application.dataPath, "csc.rsp");

        static TriLibDefineSetup()
        {
            // 도메인 리로드 한복판에서 AssetDatabase 를 만지지 않도록 지연 실행.
            EditorApplication.delayCall += () =>
            {
                MigrateAwayFromPlayerSettings();
                SyncEditor(TriLibPresent());
            };
        }

        // 어셈블리 로드 순서·이름 변형에 견고하도록 타입 조회 + 어셈블리 스캔 병행.
        private static bool TriLibPresent()
        {
            if (Type.GetType("TriLibCore.AssetLoader, TriLibCore") != null) return true;
            return AppDomain.CurrentDomain.GetAssemblies()
                .Any(a => a.GetName().Name.StartsWith("TriLibCore", StringComparison.Ordinal));
        }

        /// <summary>
        /// 과거 방식에서 PlayerSettings(ProjectSettings.asset)에 남았을 수 있는 전역
        /// TRILIB 정의를 제거한다(1회 마이그레이션). 잔존분이 없으면 아무것도 쓰지 않아
        /// ProjectSettings 를 더럽히지 않는다. TRILIB_DISABLE_EDITOR_GLTF_IMPORT 는
        /// TriLib 에디터 어셈블리가 소비하므로 그대로 둔다.
        /// </summary>
        private static void MigrateAwayFromPlayerSettings()
        {
            foreach (var target in new[] { NamedBuildTarget.Standalone, NamedBuildTarget.Android })
            {
                var list = PlayerSettings.GetScriptingDefineSymbols(target)
                    .Split(';').Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
                if (!list.Remove(TriLibDefine)) continue; // 없으면 쓰지 않음(churn 0)

                PlayerSettings.SetScriptingDefineSymbols(target, string.Join(";", list));
                UnityEngine.Debug.Log(
                    $"[TriLibDefineSetup] PlayerSettings 잔존 {TriLibDefine} 제거(target={target}) — 이제 csc.rsp 로 관리");
            }
        }

        private static void SyncEditor(bool present)
        {
            if (!EnsureRspOnDisk(present)) return; // 변경 없음 → 재컴파일 유발 안 함
            AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
            UnityEngine.Debug.Log(present
                ? $"[TriLibDefineSetup] csc.rsp 갱신 — {TriLibDefine} 정의(TriLib 설치 감지)"
                : $"[TriLibDefineSetup] csc.rsp 정리 — {TriLibDefine} 미정의(TriLib 미설치)");
        }

        /// <summary>
        /// Assets/csc.rsp 를 원하는 상태로 맞춘다(기존 사용자 옵션 라인은 보존).
        /// 디스크에 실제 변경이 발생했으면 true. 이미 원하는 상태면 false(no-op)라
        /// 도메인 리로드 루프를 만들지 않는다.
        /// </summary>
        private static bool EnsureRspOnDisk(bool present)
        {
            var lines = File.Exists(RspFullPath)
                ? File.ReadAllText(RspFullPath).Replace("\r\n", "\n").Split('\n')
                    .Select(l => l.Trim()).Where(l => l.Length > 0).ToList()
                : new List<string>();

            bool has = lines.Contains(RspLine);
            if (present == has) return false;

            if (present) lines.Add(RspLine);
            else         lines.RemoveAll(l => l == RspLine);

            if (lines.Count == 0)
            {
                if (File.Exists(RspFullPath)) File.Delete(RspFullPath);
                var meta = RspFullPath + ".meta";
                if (File.Exists(meta)) File.Delete(meta);
            }
            else
            {
                File.WriteAllText(RspFullPath, string.Join("\n", lines) + "\n");
            }
            return true;
        }

        /// <summary>
        /// 빌드 직전 csc.rsp 상태를 보장하는 안전망(배치/CI 빌드 대비).
        /// 파일만 동기화하면 이어지는 플레이어 스크립트 컴파일이 -define 을 읽어 반영한다.
        /// </summary>
        private class BuildSync : IPreprocessBuildWithReport
        {
            public int callbackOrder => 0;
            public void OnPreprocessBuild(BuildReport report) => EnsureRspOnDisk(TriLibPresent());
        }
    }
}
#endif
