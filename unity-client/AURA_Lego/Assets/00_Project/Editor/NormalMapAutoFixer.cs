using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace LegoTwin.Editor
{
    /// <summary>
    /// FBX 임포트 시 "NormalMap settings" 팝업을 자동으로 차단합니다.
    ///
    /// 대상 텍스처 판별 기준 (대소문자 무시):
    ///   - 파일명이 "normal" 또는 "bump"로 시작  (예: NormalGL_xxxx.jpg, Normal_body.png)
    ///   - 파일명이 _normal/_nrm/_n/_bump/_nor 로 끝남
    ///
    /// 동작 흐름:
    ///   OnPreprocessTexture — 텍스처 임포트 시점에 즉시 NormalMap 타입 지정
    ///                         (FBX 임베디드 텍스처도 .fbm 추출 시 이 콜백을 탄다)
    ///   OnPreprocessModel   — FBX 재임포트 시 .fbm 폴더에 이미 있는 텍스처를 사전 수정
    /// </summary>
    public class NormalMapAutoFixer : AssetPostprocessor
    {
        private static readonly string[] EndSuffixes =
            { "_normal", "_nrm", "_n", "_bump", "_nor" };

        private static readonly string[] StartPrefixes =
            { "normal", "bump" };

        // ── 텍스처 임포트 시: 파일명으로 판별해 즉시 NormalMap 타입 지정 ──────────
        // FBX 임베디드 텍스처는 .fbm 폴더 추출 시 이 콜백이 호출된다.
        void OnPreprocessTexture()
        {
            if (!IsNormalMapName(Path.GetFileNameWithoutExtension(assetPath))) return;

            var importer = (TextureImporter)assetImporter;
            if (importer.textureType == TextureImporterType.NormalMap) return;

            importer.textureType = TextureImporterType.NormalMap;
            Debug.Log($"[NormalMapAutoFixer] NormalMap 자동 지정: {assetPath}");
        }

        // ── FBX 임포트 시작 전: .fbm 폴더 포함해 기존 텍스처 사전 수정 ────────────
        // 재임포트 시 .fbm 폴더가 이미 존재하므로 여기서 사전 처리 가능.
        void OnPreprocessModel()
        {
            string dir = Path.GetDirectoryName(assetPath)?.Replace("\\", "/");
            if (string.IsNullOrEmpty(dir)) return;

            // 탐색 대상: 같은 폴더 / .fbm 폴더 / Textures 하위 폴더
            var searchFolders = new List<string> { dir };

            string fbmFolder = Path.ChangeExtension(assetPath, null)?.Replace("\\", "/") + ".fbm";
            if (AssetDatabase.IsValidFolder(fbmFolder))
                searchFolders.Add(fbmFolder);

            string texFolder = dir + "/Textures";
            if (AssetDatabase.IsValidFolder(texFolder))
                searchFolders.Add(texFolder);

            string[] guids = AssetDatabase.FindAssets("t:Texture2D", searchFolders.ToArray());
            foreach (string guid in guids)
            {
                string texPath = AssetDatabase.GUIDToAssetPath(guid);
                if (!IsNormalMapName(Path.GetFileNameWithoutExtension(texPath))) continue;

                var texImporter = AssetImporter.GetAtPath(texPath) as TextureImporter;
                if (texImporter == null || texImporter.textureType == TextureImporterType.NormalMap) continue;

                texImporter.textureType = TextureImporterType.NormalMap;
                texImporter.SaveAndReimport();
                Debug.Log($"[NormalMapAutoFixer] NormalMap 사전 수정 (재임포트): {texPath}");
            }
        }

        static bool IsNormalMapName(string filename)
        {
            string lower = filename.ToLower();

            // "NormalGL_xxxx", "Normal_body", "Bump_xxx" 등 접두어 패턴
            foreach (string prefix in StartPrefixes)
                if (lower.StartsWith(prefix)) return true;

            // "_normal", "_nrm", "_n", "_bump", "_nor" 접미어 패턴
            foreach (string suffix in EndSuffixes)
                if (lower.EndsWith(suffix)) return true;

            return false;
        }
    }
}
