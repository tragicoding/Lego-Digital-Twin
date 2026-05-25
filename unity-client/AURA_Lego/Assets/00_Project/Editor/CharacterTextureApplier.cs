using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace LegoTwin.EditorTools
{
    /// <summary>
    /// GeneratedCharacters/**/character/ 폴더의 FBX+GLB 쌍을 감지하여
    /// GLB 의 텍스쳐를 FBX 머티리얼에 자동 적용하는 에디터 스크립트.
    ///
    /// ── 자동 실행 ──────────────────────────────────────────────────
    /// FBX(*_rigged.fbx) 또는 GLB(*_texture.glb) 를 폴더에 넣으면
    /// 페어 파일이 존재할 때 자동으로 텍스쳐를 적용합니다.
    ///
    /// ── 수동 실행 ──────────────────────────────────────────────────
    /// Unity 메뉴: Tools > MINIVERSE > Apply Character Textures
    ///
    /// ── 처리 흐름 ──────────────────────────────────────────────────
    ///  1. GLB 에서 Texture2D 및 Material 추출
    ///     (glTFast 6.x 가 GLB를 임포트하면 sub-asset으로 접근 가능)
    ///  2. URP Lit 머티리얼(.mat) 생성 / 갱신
    ///     저장 위치: GeneratedCharacters/**/character/Materials/{npc_N}.mat
    ///  3. FBX ModelImporter → External 머티리얼 모드로 전환
    ///  4. FBX 내 모든 머티리얼을 위의 .mat 으로 리맵 후 SaveAndReimport
    /// </summary>
    public class CharacterTextureApplier : AssetPostprocessor
    {
        private const string CHAR_FOLDER = "GeneratedCharacters";
        private const string FBX_SUFFIX  = "_rigged.fbx";
        private const string GLB_SUFFIX  = "_texture.glb";

        // 재귀 임포트 방지 플래그
        private static bool _isApplying = false;

        // ════════════════════════════════════════════════════════════
        // 자동 감지 — 임포트 후 훅
        // ════════════════════════════════════════════════════════════

        static void OnPostprocessAllAssets(
            string[] imported, string[] deleted,
            string[] moved,    string[] movedFrom)
        {
            if (_isApplying) return;

            foreach (var path in imported)
            {
                if (!path.Contains(CHAR_FOLDER)) continue;
                if (!path.Contains("/character/")) continue;

                if (path.EndsWith(GLB_SUFFIX, System.StringComparison.OrdinalIgnoreCase))
                {
                    // GLB 임포트 완료 → 페어 FBX 에 적용
                    string fbx = path.Replace(GLB_SUFFIX, FBX_SUFFIX);
                    if (FileExists(fbx))
                        EditorApplication.delayCall += () => ApplyGlbTextureToFbx(fbx, path);
                }
                else if (path.EndsWith(FBX_SUFFIX, System.StringComparison.OrdinalIgnoreCase))
                {
                    // FBX 임포트 완료 → GLB 가 이미 존재하면 적용
                    string glb = path.Replace(FBX_SUFFIX, GLB_SUFFIX);
                    if (FileExists(glb) && AssetDatabase.LoadAssetAtPath<Object>(glb) != null)
                        EditorApplication.delayCall += () => ApplyGlbTextureToFbx(path, glb);
                }
            }
        }

        // ════════════════════════════════════════════════════════════
        // 수동 실행 — 메뉴
        // ════════════════════════════════════════════════════════════

        [MenuItem("Tools/MINIVERSE/Apply Character Textures")]
        public static void MenuApplyAll()
        {
            string searchRoot = Path.Combine(
                Application.dataPath, "00_Project", CHAR_FOLDER);

            if (!Directory.Exists(searchRoot))
            {
                EditorUtility.DisplayDialog("MINIVERSE",
                    $"폴더 없음: {searchRoot}", "확인");
                return;
            }

            var fbxFiles = Directory.GetFiles(searchRoot, $"*{FBX_SUFFIX}",
                                              SearchOption.AllDirectories);
            int count = 0;
            foreach (var absPath in fbxFiles)
            {
                // 절대 경로 → Unity 에셋 경로
                string fbxPath = ToAssetPath(absPath);
                string glbPath = fbxPath.Replace(FBX_SUFFIX, GLB_SUFFIX);

                if (AssetDatabase.LoadAssetAtPath<Object>(glbPath) == null)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] GLB 미임포트 — 건너뜀: {glbPath}");
                    continue;
                }

                ApplyGlbTextureToFbx(fbxPath, glbPath);
                count++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[CharacterTextureApplier] ✅ 완료 — {count}개 처리");
            EditorUtility.DisplayDialog("MINIVERSE",
                $"캐릭터 텍스쳐 적용 완료\n처리된 캐릭터: {count}개", "확인");
        }

        // ════════════════════════════════════════════════════════════
        // 핵심 로직
        // ════════════════════════════════════════════════════════════

        static void ApplyGlbTextureToFbx(string fbxPath, string glbPath)
        {
            if (_isApplying) return;
            if (!FileExists(fbxPath) || !FileExists(glbPath)) return;

            _isApplying = true;
            try
            {
                Debug.Log($"[CharacterTextureApplier] 처리 중: " +
                          $"{Path.GetFileName(fbxPath)} ← {Path.GetFileName(glbPath)}");

                // ── 1. GLB 에서 텍스쳐 추출 ──────────────────────────
                var glbSubs  = AssetDatabase.LoadAllAssetsAtPath(glbPath);
                var glbMat   = glbSubs.OfType<Material>().FirstOrDefault();
                var textures = glbSubs.OfType<Texture2D>().ToArray();

                if (glbSubs.Length == 0)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] GLB sub-asset 없음 — " +
                        $"glTFast 임포트 완료 후 다시 시도하세요: {glbPath}");
                    return;
                }

                // 우선순위: GLB Material 프로퍼티 → Texture2D 이름 패턴 → 첫 번째 텍스쳐
                Texture2D baseColorTex = FindTex(glbMat, textures,
                    new[] { "_BaseMap", "_BaseColorMap", "_MainTex" },
                    new[] { "basecolor", "albedo", "diffuse", "color" });

                Texture2D normalTex = FindTex(glbMat, textures,
                    new[] { "_BumpMap", "_NormalMap" },
                    new[] { "normal", "nrm", "nrml" });

                Texture2D metallicTex = FindTex(glbMat, textures,
                    new[] { "_MetallicGlossMap", "_MetallicRoughnessMap" },
                    new[] { "metallic", "roughness", "orm", "metallicroughness" });

                Texture2D emissiveTex = FindTex(glbMat, textures,
                    new[] { "_EmissionMap", "_EmissiveColorMap" },
                    new[] { "emissive", "emission" });

                // baseColor 를 못 찾으면 첫 번째 Texture2D 사용
                if (baseColorTex == null && textures.Length > 0)
                    baseColorTex = textures[0];

                if (baseColorTex == null)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] Texture2D 없음: {glbPath}\n" +
                        $"GLB sub-asset 수: {glbSubs.Length}");
                    return;
                }

                // ── 2. Materials 폴더에 .mat 생성 / 갱신 ────────────
                string fbxDir  = Path.GetDirectoryName(fbxPath);
                string matDir  = fbxDir + "/Materials";
                if (!AssetDatabase.IsValidFolder(matDir))
                    AssetDatabase.CreateFolder(fbxDir, "Materials");

                // mat 이름: npc_3_rigged → npc_3
                string baseName = Path.GetFileNameWithoutExtension(fbxPath)
                                     .Replace("_rigged", "");
                string matPath  = $"{matDir}/{baseName}.mat";

                Material mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
                if (mat == null)
                {
                    mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                    AssetDatabase.CreateAsset(mat, matPath);
                }

                // 텍스쳐 적용
                mat.SetTexture("_BaseMap", baseColorTex);
                mat.SetColor("_BaseColor", glbMat != null ? glbMat.color : Color.white);

                if (normalTex != null)
                {
                    mat.SetTexture("_BumpMap", normalTex);
                    mat.EnableKeyword("_NORMALMAP");
                }
                if (metallicTex != null)
                {
                    mat.SetTexture("_MetallicGlossMap", metallicTex);
                    mat.EnableKeyword("_METALLICSPECGLOSSMAP");
                }
                if (emissiveTex != null)
                {
                    mat.SetTexture("_EmissionMap", emissiveTex);
                    mat.EnableKeyword("_EMISSION");
                    mat.globalIlluminationFlags =
                        MaterialGlobalIlluminationFlags.RealtimeEmissive;
                }

                EditorUtility.SetDirty(mat);
                AssetDatabase.SaveAssets();

                // ── 3. FBX → External 머티리얼 모드 + 리맵 ──────────
                var importer = AssetImporter.GetAtPath(fbxPath) as ModelImporter;
                if (importer == null)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] ModelImporter 없음: {fbxPath}");
                    return;
                }

                importer.materialImportMode =
                    ModelImporterMaterialImportMode.ImportViaMaterialDescription;
                importer.materialLocation =
                    ModelImporterMaterialLocation.External;

                // FBX 내 embedded 머티리얼 이름 수집
                var srcMats = AssetDatabase.LoadAllAssetsAtPath(fbxPath)
                                           .OfType<Material>().ToList();

                if (srcMats.Count > 0)
                {
                    foreach (var srcMat in srcMats)
                        importer.AddRemap(
                            new AssetImporter.SourceAssetIdentifier(
                                typeof(Material), srcMat.name), mat);
                }
                else
                {
                    // embedded 머티리얼이 아직 없으면 기본 이름 "Material" 로 리맵
                    importer.AddRemap(
                        new AssetImporter.SourceAssetIdentifier(
                            typeof(Material), "Material"), mat);
                }

                importer.SaveAndReimport();

                Debug.Log(
                    $"[CharacterTextureApplier] ✅ 적용 완료: {Path.GetFileName(fbxPath)}\n" +
                    $"  BaseColor : {baseColorTex.name}\n" +
                    $"  Normal    : {(normalTex    != null ? normalTex.name    : "없음")}\n" +
                    $"  Metallic  : {(metallicTex  != null ? metallicTex.name  : "없음")}\n" +
                    $"  Emissive  : {(emissiveTex  != null ? emissiveTex.name  : "없음")}");
            }
            finally
            {
                _isApplying = false;
            }
        }

        // ════════════════════════════════════════════════════════════
        // 유틸리티
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// Material 프로퍼티 → Texture2D 이름 패턴 순서로 텍스쳐를 탐색.
        /// </summary>
        static Texture2D FindTex(
            Material glbMat,
            Texture2D[] pool,
            string[] matProps,
            string[] nameKeywords)
        {
            // 1. GLB Material 프로퍼티에서 탐색
            if (glbMat != null)
            {
                foreach (var prop in matProps)
                {
                    if (!glbMat.HasProperty(prop)) continue;
                    var t = glbMat.GetTexture(prop) as Texture2D;
                    if (t != null) return t;
                }
            }
            // 2. Texture2D 이름 키워드로 탐색
            foreach (var kw in nameKeywords)
                foreach (var t in pool)
                    if (t.name.ToLower().Contains(kw)) return t;

            return null;
        }

        /// <summary>Unity 에셋 경로 → 파일시스템 절대 경로 존재 여부 확인.</summary>
        static bool FileExists(string assetPath) =>
            File.Exists(Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", assetPath)));

        /// <summary>파일시스템 절대 경로 → Unity 에셋 경로 변환.</summary>
        static string ToAssetPath(string absPath)
        {
            absPath = absPath.Replace('\\', '/');
            string dataPath = Application.dataPath.Replace('\\', '/');
            // dataPath = "...project/Assets"
            int idx = absPath.IndexOf("/Assets/");
            return idx >= 0 ? absPath.Substring(idx + 1) : absPath;
        }
    }
}
