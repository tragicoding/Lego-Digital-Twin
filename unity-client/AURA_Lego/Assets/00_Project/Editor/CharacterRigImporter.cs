using UnityEditor;
using UnityEngine;

namespace LegoTwin.EditorTools
{
    /// <summary>
    /// GeneratedCharacters/**/character/ 폴더의 캐릭터 .fbx 임포트 시 자동 처리:
    ///   1. Animation Type → Humanoid
    ///   2. Animator Controller 자동 연결
    ///      (Assets/00_Project/Animations/Character/NPC_AnimatorController.controller)
    ///
    /// 대상 명명: npc_X_rigged.fbx(기존) 와 Character_N/Character_N.fbx(서버 포맷) 둘 다 포함.
    ///           (.fbm 임베디드 텍스처 폴더 안의 파일은 제외)
    ///
    /// ── 수동 일괄 적용 ─────────────────────────────────────────────
    ///  Tools > MINIVERSE > Set Characters Humanoid
    ///  Tools > MINIVERSE > Connect Character Animator
    /// </summary>
    public class CharacterRigImporter : AssetPostprocessor
    {
        private const string CHAR_FOLDER      = "GeneratedCharacters";
        private const string FBX_SUFFIX       = ".fbx";
        private const string CONTROLLER_PATH  =
            "Assets/00_Project/Animations/Character/NPC_AnimatorController.controller";

        // ════════════════════════════════════════════════════════════
        // 임포트 전 — Humanoid 설정
        // ════════════════════════════════════════════════════════════

        void OnPreprocessModel()
        {
            if (!IsTarget(assetPath)) return;

            var importer = (ModelImporter)assetImporter;
            if (importer.animationType == ModelImporterAnimationType.Human) return;

            importer.animationType = ModelImporterAnimationType.Human;
            Debug.Log($"[CharacterRigImporter] Humanoid 자동 적용: {assetPath}");
        }

        // ════════════════════════════════════════════════════════════
        // 임포트 후 — Animator Controller 자동 연결
        // ════════════════════════════════════════════════════════════

        void OnPostprocessModel(GameObject go)
        {
            if (!IsTarget(assetPath)) return;

            var animator = go.GetComponentInChildren<Animator>(true);
            if (animator == null) return;
            if (animator.runtimeAnimatorController != null) return; // 이미 연결됨

            var controller = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>(CONTROLLER_PATH);
            if (controller == null)
            {
                Debug.Log($"[CharacterRigImporter] Animator Controller 없음 — 생성 후 재임포트하거나 수동 연결:\n{CONTROLLER_PATH}");
                return;
            }

            animator.runtimeAnimatorController = controller;
            Debug.Log($"[CharacterRigImporter] Animator Controller 자동 연결: {System.IO.Path.GetFileName(assetPath)}");
        }

        // ════════════════════════════════════════════════════════════
        // 수동 일괄 적용 — 메뉴
        // ════════════════════════════════════════════════════════════

        [MenuItem("Tools/MINIVERSE/Set Characters Humanoid")]
        public static void MenuSetAllHumanoid()
        {
            int count = ProcessAll((path, importer) =>
            {
                if (importer.animationType == ModelImporterAnimationType.Human) return false;
                importer.animationType = ModelImporterAnimationType.Human;
                Debug.Log($"[CharacterRigImporter] Humanoid 적용: {path}");
                return true;
            });

            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("MINIVERSE", $"Humanoid 설정 완료\n처리: {count}개", "확인");
        }

        [MenuItem("Tools/MINIVERSE/Connect Character Animator")]
        public static void MenuConnectAnimator()
        {
            var controller = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>(CONTROLLER_PATH);
            if (controller == null)
            {
                EditorUtility.DisplayDialog("MINIVERSE",
                    $"Animator Controller 없음:\n{CONTROLLER_PATH}\n\n" +
                    "먼저 해당 경로에 Animator Controller를 생성하세요.", "확인");
                return;
            }

            int count = 0;
            var guids = AssetDatabase.FindAssets("t:Model", new[] { "Assets/00_Project/GeneratedCharacters" });
            foreach (var guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (!IsTarget(path)) continue;

                // 임포트된 모델의 루트 오브젝트 로드
                var root = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (root == null) continue;

                var animator = root.GetComponentInChildren<Animator>(true);
                if (animator == null || animator.runtimeAnimatorController != null) continue;

                // SerializedObject 를 통해 수정해야 에셋에 저장됨
                using var so = new SerializedObject(animator);
                var prop = so.FindProperty("m_Controller");
                prop.objectReferenceValue = controller;
                so.ApplyModifiedPropertiesWithoutUndo();

                count++;
                Debug.Log($"[CharacterRigImporter] Animator Controller 연결: {path}");
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("MINIVERSE",
                $"Animator Controller 연결 완료\n처리: {count}개", "확인");
        }

        // ════════════════════════════════════════════════════════════
        // 유틸
        // ════════════════════════════════════════════════════════════

        private static bool IsTarget(string path) =>
            path.Contains(CHAR_FOLDER) &&
            path.Contains("/character/") &&
            !path.Contains(".fbm/") &&   // .fbm = 임베디드 텍스처 폴더 — 모델 아님, 제외
            path.EndsWith(FBX_SUFFIX, System.StringComparison.OrdinalIgnoreCase);

        private static int ProcessAll(System.Func<string, ModelImporter, bool> action)
        {
            int count = 0;
            var guids = AssetDatabase.FindAssets("t:Model", new[] { "Assets/00_Project/GeneratedCharacters" });
            foreach (var guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (!IsTarget(path)) continue;

                var importer = AssetImporter.GetAtPath(path) as ModelImporter;
                if (importer == null) continue;

                if (action(path, importer))
                {
                    importer.SaveAndReimport();
                    count++;
                }
            }
            return count;
        }
    }
}
