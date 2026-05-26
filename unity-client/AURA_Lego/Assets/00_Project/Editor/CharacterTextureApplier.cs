using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace LegoTwin.EditorTools
{
    /// <summary>
    /// GeneratedCharacters/**/character/ 폴더의 FBX+GLB 쌍을 감지하여
    /// GLB 안에 포함된 Texture2D를 새 Material의 BaseMap/MainTex에 연결하고,
    /// FBX 캐릭터의 SkinnedMeshRenderer에 자동 적용하는 에디터 스크립트.
    ///
    /// 핵심 변경점:
    /// - GLB Material에서 텍스처를 추출하지 않음
    /// - GLB sub-asset 중 Texture2D를 직접 찾음
    /// - 완전히 새로운 Material 생성
    /// - FBX의 SkinnedMeshRenderer에 새 Material 적용
    /// - 원본 FBX rig / bone / Animator 구조는 변경하지 않음
    /// - 적용된 상태의 Prefab을 GeneratedCharacters/**/character/Prefabs/ 에 저장
    /// </summary>
    public class CharacterTextureApplier : AssetPostprocessor
    {
        private const string CHAR_FOLDER = "GeneratedCharacters";
        private const string FBX_SUFFIX = "_rigged.fbx";
        private const string GLB_SUFFIX = "_texture.glb";

        private static bool _isApplying = false;

        // ════════════════════════════════════════════════════════════
        // 자동 감지 — 임포트 후 훅
        // ════════════════════════════════════════════════════════════

        static void OnPostprocessAllAssets(
            string[] imported,
            string[] deleted,
            string[] moved,
            string[] movedFrom)
        {
            if (_isApplying) return;

            foreach (var path in imported)
            {
                if (!path.Contains(CHAR_FOLDER)) continue;
                if (!path.Contains("/character/")) continue;

                if (path.EndsWith(GLB_SUFFIX, System.StringComparison.OrdinalIgnoreCase))
                {
                    string fbxPath = path.Replace(GLB_SUFFIX, FBX_SUFFIX);

                    if (FileExists(fbxPath))
                    {
                        EditorApplication.delayCall += () =>
                            ApplyGlbTexture2DToFbx(fbxPath, path);
                    }
                }
                else if (path.EndsWith(FBX_SUFFIX, System.StringComparison.OrdinalIgnoreCase))
                {
                    string glbPath = path.Replace(FBX_SUFFIX, GLB_SUFFIX);

                    if (FileExists(glbPath) &&
                        AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(glbPath) != null)
                    {
                        EditorApplication.delayCall += () =>
                            ApplyGlbTexture2DToFbx(path, glbPath);
                    }
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
                Application.dataPath,
                "00_Project",
                CHAR_FOLDER
            );

            if (!Directory.Exists(searchRoot))
            {
                EditorUtility.DisplayDialog(
                    "MINIVERSE",
                    $"폴더 없음: {searchRoot}",
                    "확인"
                );
                return;
            }

            var fbxFiles = Directory.GetFiles(
                searchRoot,
                $"*{FBX_SUFFIX}",
                SearchOption.AllDirectories
            );

            int count = 0;

            foreach (var absPath in fbxFiles)
            {
                string fbxPath = ToAssetPath(absPath);
                string glbPath = fbxPath.Replace(FBX_SUFFIX, GLB_SUFFIX);

                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(glbPath) == null)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] GLB 미임포트 또는 없음 — 건너뜀: {glbPath}"
                    );
                    continue;
                }

                ApplyGlbTexture2DToFbx(fbxPath, glbPath);
                count++;
            }

            AssetDatabase.Refresh();

            Debug.Log($"[CharacterTextureApplier] ✅ 완료 — {count}개 처리");

            EditorUtility.DisplayDialog(
                "MINIVERSE",
                $"캐릭터 텍스쳐 적용 완료\n처리된 캐릭터: {count}개",
                "확인"
            );
        }

        // ════════════════════════════════════════════════════════════
        // 핵심 로직
        // ════════════════════════════════════════════════════════════

        private static void ApplyGlbTexture2DToFbx(string fbxPath, string glbPath)
        {
            if (_isApplying) return;
            if (!FileExists(fbxPath) || !FileExists(glbPath)) return;

            _isApplying = true;

            try
            {
                Debug.Log(
                    $"[CharacterTextureApplier] 처리 시작: " +
                    $"{Path.GetFileName(fbxPath)} ← {Path.GetFileName(glbPath)}"
                );

                // 1. GLB sub-asset에서 Texture2D 직접 찾기
                Texture2D texture = ExtractTexture2DFromGlb(glbPath);

                if (texture == null)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] GLB에서 Texture2D를 찾지 못했습니다: {glbPath}"
                    );
                    return;
                }

                // 2. Texture2D 기반 새 Material 생성 또는 갱신
                Material generatedMaterial = CreateOrUpdateMaterialFromTexture(
                    fbxPath,
                    texture
                );

                if (generatedMaterial == null)
                {
                    Debug.LogWarning(
                        $"[CharacterTextureApplier] Material 생성 실패: {fbxPath}"
                    );
                    return;
                }

                // 3. FBX Importer의 Material Remap에도 적용
                //    FBX asset 자체를 Project 창에서 열어도 material이 연결되게 하기 위함
                ApplyMaterialRemapToFbx(fbxPath, generatedMaterial);

                // 4. FBX를 instantiate해서 SkinnedMeshRenderer에 새 Material 직접 적용
                //    실제 Unity 개발자가 사용할 Prefab 생성
                CreateOrUpdateTexturedPrefab(fbxPath, generatedMaterial);

                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                Debug.Log(
                    $"[CharacterTextureApplier] ✅ 적용 완료\n" +
                    $"  FBX      : {fbxPath}\n" +
                    $"  GLB      : {glbPath}\n" +
                    $"  Texture  : {texture.name}\n" +
                    $"  Material : {AssetDatabase.GetAssetPath(generatedMaterial)}"
                );
            }
            finally
            {
                _isApplying = false;
            }
        }

        /// <summary>
        /// GLB sub-asset 중 Texture2D를 직접 찾는다.
        /// Material 프로퍼티는 사용하지 않는다.
        /// </summary>
        private static Texture2D ExtractTexture2DFromGlb(string glbPath)
        {
            var subAssets = AssetDatabase.LoadAllAssetsAtPath(glbPath);

            if (subAssets == null || subAssets.Length == 0)
            {
                Debug.LogWarning(
                    $"[CharacterTextureApplier] GLB sub-asset 없음: {glbPath}"
                );
                return null;
            }

            var textures = subAssets
                .OfType<Texture2D>()
                .Where(t => t != null)
                .ToArray();

            if (textures.Length == 0)
            {
                Debug.LogWarning(
                    $"[CharacterTextureApplier] Texture2D sub-asset 없음: {glbPath}"
                );
                return null;
            }

            Debug.Log(
                $"[CharacterTextureApplier] GLB Texture2D 후보 {textures.Length}개 발견:\n" +
                string.Join("\n", textures.Select(t => $"  - {t.name} ({t.width}x{t.height})"))
            );

            // 이름 기준 우선순위
            string[] baseColorKeywords =
            {
                "basecolor",
                "base_color",
                "albedo",
                "diffuse",
                "color",
                "texture",
                "tex"
            };

            foreach (var keyword in baseColorKeywords)
            {
                Texture2D matched = textures.FirstOrDefault(
                    t => t.name.ToLower().Contains(keyword)
                );

                if (matched != null)
                {
                    Debug.Log(
                        $"[CharacterTextureApplier] Texture2D 선택: {matched.name} / keyword={keyword}"
                    );
                    return matched;
                }
            }

            // 못 찾으면 첫 번째 Texture2D 사용
            Debug.LogWarning(
                $"[CharacterTextureApplier] 이름 패턴으로 BaseColor를 찾지 못해 첫 번째 Texture2D 사용: {textures[0].name}"
            );

            return textures[0];
        }

        /// <summary>
        /// Texture2D를 BaseMap/MainTex로 사용하는 새 Material 생성 또는 갱신.
        /// </summary>
        private static Material CreateOrUpdateMaterialFromTexture(
            string fbxPath,
            Texture2D texture)
        {
            string fbxDir = Path.GetDirectoryName(fbxPath);
            string matDir = fbxDir + "/Materials";

            if (!AssetDatabase.IsValidFolder(matDir))
            {
                AssetDatabase.CreateFolder(fbxDir, "Materials");
            }

            string baseName = Path.GetFileNameWithoutExtension(fbxPath)
                .Replace("_rigged", "");

            string matPath = $"{matDir}/{baseName}_TextureFromGLB.mat";

            Material mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);

            if (mat == null)
            {
                Shader shader = FindBestShader();

                if (shader == null)
                {
                    Debug.LogWarning(
                        "[CharacterTextureApplier] 사용할 수 있는 Shader를 찾지 못했습니다."
                    );
                    return null;
                }

                mat = new Material(shader)
                {
                    name = $"{baseName}_TextureFromGLB"
                };

                AssetDatabase.CreateAsset(mat, matPath);

                Debug.Log(
                    $"[CharacterTextureApplier] 새 Material 생성: {matPath}"
                );
            }

            ApplyTextureToMaterial(mat, texture);

            EditorUtility.SetDirty(mat);

            return mat;
        }

        private static Shader FindBestShader()
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");

            if (shader != null)
            {
                return shader;
            }

            shader = Shader.Find("Standard");

            if (shader != null)
            {
                return shader;
            }

            shader = Shader.Find("Unlit/Texture");

            if (shader != null)
            {
                return shader;
            }

            return null;
        }

        private static void ApplyTextureToMaterial(Material mat, Texture2D texture)
        {
            if (mat == null || texture == null) return;

            if (mat.HasProperty("_BaseMap"))
            {
                mat.SetTexture("_BaseMap", texture);
            }

            if (mat.HasProperty("_MainTex"))
            {
                mat.SetTexture("_MainTex", texture);
            }

            if (mat.HasProperty("_BaseColor"))
            {
                mat.SetColor("_BaseColor", Color.white);
            }

            if (mat.HasProperty("_Color"))
            {
                mat.SetColor("_Color", Color.white);
            }

            mat.mainTexture = texture;

            Debug.Log(
                $"[CharacterTextureApplier] Material에 Texture2D 연결 완료: {mat.name} ← {texture.name}"
            );
        }

        /// <summary>
        /// FBX importer의 material remap 설정.
        /// Project 창의 FBX asset 자체에도 material 연결이 되도록 한다.
        /// </summary>
        private static void ApplyMaterialRemapToFbx(
            string fbxPath,
            Material generatedMaterial)
        {
            var importer = AssetImporter.GetAtPath(fbxPath) as ModelImporter;

            if (importer == null)
            {
                Debug.LogWarning(
                    $"[CharacterTextureApplier] ModelImporter 없음: {fbxPath}"
                );
                return;
            }

            importer.materialImportMode =
                ModelImporterMaterialImportMode.ImportViaMaterialDescription;

            importer.materialLocation =
                ModelImporterMaterialLocation.External;

            var sourceMaterials = AssetDatabase
                .LoadAllAssetsAtPath(fbxPath)
                .OfType<Material>()
                .ToList();

            if (sourceMaterials.Count > 0)
            {
                foreach (var srcMat in sourceMaterials)
                {
                    importer.AddRemap(
                        new AssetImporter.SourceAssetIdentifier(
                            typeof(Material),
                            srcMat.name
                        ),
                        generatedMaterial
                    );

                    Debug.Log(
                        $"[CharacterTextureApplier] Material Remap: {srcMat.name} → {generatedMaterial.name}"
                    );
                }
            }
            else
            {
                importer.AddRemap(
                    new AssetImporter.SourceAssetIdentifier(
                        typeof(Material),
                        "Material"
                    ),
                    generatedMaterial
                );

                Debug.LogWarning(
                    $"[CharacterTextureApplier] FBX source material을 찾지 못해 기본 이름 'Material'에 remap 적용"
                );
            }

            importer.SaveAndReimport();
        }

        /// <summary>
        /// FBX prefab을 instantiate한 뒤 SkinnedMeshRenderer에 새 Material을 직접 적용하고,
        /// 적용된 상태의 Prefab을 Generated Prefabs 폴더에 저장한다.
        /// </summary>
        private static void CreateOrUpdateTexturedPrefab(
            string fbxPath,
            Material generatedMaterial)
        {
            GameObject fbxAsset = AssetDatabase.LoadAssetAtPath<GameObject>(fbxPath);

            if (fbxAsset == null)
            {
                Debug.LogWarning(
                    $"[CharacterTextureApplier] FBX GameObject 로드 실패: {fbxPath}"
                );
                return;
            }

            GameObject instance = null;

            try
            {
                instance = (GameObject)PrefabUtility.InstantiatePrefab(fbxAsset);

                if (instance == null)
                {
                    instance = Object.Instantiate(fbxAsset);
                }

                string baseName = Path.GetFileNameWithoutExtension(fbxPath)
                    .Replace("_rigged", "");

                instance.name = $"{baseName}_Textured";

                ApplyMaterialToSkinnedMeshRenderers(instance, generatedMaterial);

                string fbxDir = Path.GetDirectoryName(fbxPath);
                string prefabDir = fbxDir + "/Prefabs";

                if (!AssetDatabase.IsValidFolder(prefabDir))
                {
                    AssetDatabase.CreateFolder(fbxDir, "Prefabs");
                }

                string prefabPath = $"{prefabDir}/{baseName}_Textured.prefab";

                PrefabUtility.SaveAsPrefabAsset(instance, prefabPath);

                Debug.Log(
                    $"[CharacterTextureApplier] Textured Prefab 저장 완료: {prefabPath}"
                );
            }
            finally
            {
                if (instance != null)
                {
                    Object.DestroyImmediate(instance);
                }
            }
        }

        /// <summary>
        /// GameObject 하위의 모든 SkinnedMeshRenderer에 Material 적용.
        /// FBX의 bone / rig / Animator는 건드리지 않는다.
        /// </summary>
        private static void ApplyMaterialToSkinnedMeshRenderers(
            GameObject root,
            Material material)
        {
            if (root == null)
            {
                Debug.LogWarning(
                    "[CharacterTextureApplier] root GameObject가 null입니다."
                );
                return;
            }

            if (material == null)
            {
                Debug.LogWarning(
                    "[CharacterTextureApplier] 적용할 Material이 null입니다."
                );
                return;
            }

            var skinnedRenderers =
                root.GetComponentsInChildren<SkinnedMeshRenderer>(true);

            if (skinnedRenderers == null || skinnedRenderers.Length == 0)
            {
                Debug.LogWarning(
                    $"[CharacterTextureApplier] SkinnedMeshRenderer 없음: {root.name}"
                );
                return;
            }

            foreach (var renderer in skinnedRenderers)
            {
                Material[] before = renderer.sharedMaterials;

                int slotCount = Mathf.Max(1, before.Length);
                Material[] newMaterials = new Material[slotCount];

                for (int i = 0; i < slotCount; i++)
                {
                    newMaterials[i] = material;
                }

                renderer.sharedMaterials = newMaterials;

                Debug.Log(
                    $"[CharacterTextureApplier] SkinnedMeshRenderer Material 적용\n" +
                    $"  Renderer : {renderer.name}\n" +
                    $"  SlotCount: {slotCount}\n" +
                    $"  NewMat   : {material.name}"
                );
            }
        }

        // ════════════════════════════════════════════════════════════
        // 유틸리티
        // ════════════════════════════════════════════════════════════

        private static bool FileExists(string assetPath)
        {
            string absolutePath = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", assetPath)
            );

            return File.Exists(absolutePath);
        }

        private static string ToAssetPath(string absPath)
        {
            absPath = absPath.Replace('\\', '/');

            int idx = absPath.IndexOf("/Assets/");

            return idx >= 0 ? absPath.Substring(idx + 1) : absPath;
        }
    }
}