using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 캐릭터 스포너.
    /// 같은 캐릭터를 두 역할로 생성한다:
    ///   1. Guide  — 가이드 NPC (걸어다니며 안내)
    ///   2. Placed — 광장 배치 캐릭터 (오브제 옆, 모션 수행)
    ///
    /// Mock Mode : Inspector에 연결된 mockCharacterPrefab을 Instantiate.
    ///             GeneratedCharacters/mocking/character/ 에 FBX를 넣고
    ///             Unity가 임포트하면 Prefab 생성 → 이 필드에 드래그.
    ///
    /// Server Mode: model_url(FBX) + texture_url(GLB)을 다운로드해 런타임 로드.
    ///              TODO: ModelLoader 구현 후 연동 (TriLib 또는 glTFast + FBX 파서)
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] mockCharacterPrefab 에 mocking/character/ FBX로 만든 Prefab 연결
    ///   [ ] guideSpawnPoint, placedSpawnPoint 위치 지정
    ///   [ ] Server Mode 용 SpawnFromServer() 내 TODO 완성
    /// </summary>
    public class GeneratedCharacterSpawner : MonoBehaviour
    {
        [Header("Mock Mode — mocking/character/ 의 FBX로 만든 Prefab 연결")]
        public GameObject mockCharacterPrefab;

        [Header("스폰 위치")]
        [Tooltip("가이드 NPC 시작 위치")]
        public Transform guideSpawnPoint;
        [Tooltip("광장 배치 캐릭터 위치 (오브제 옆)")]
        public Transform placedSpawnPoint;

        private GameObject _guideInstance;
        private GameObject _placedInstance;

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 가이드 NPC 생성 후 GuideNPCController 반환.
        /// SessionManager.OnSessionLoaded 콜백에서 호출.
        /// </summary>
        public GuideNPCController SpawnGuide(SessionData session)
        {
            if (_guideInstance != null) Destroy(_guideInstance);

            var pos = guideSpawnPoint  != null ? guideSpawnPoint.position  : Vector3.zero;
            var rot = guideSpawnPoint  != null ? guideSpawnPoint.rotation  : Quaternion.identity;

            _guideInstance = SpawnCharacter(session.assets?.character, pos, rot, "Guide");
            if (_guideInstance == null) return null;

            var npc = _guideInstance.GetComponent<GuideNPCController>();
            npc?.Initialize(session);
            return npc;
        }

        /// <summary>
        /// 광장 배치 캐릭터 생성 (오브제 옆).
        /// 가이드 NPC와 동일 모델, 다른 위치.
        /// 모션/리타겟팅은 반환된 GameObject에서 직접 제어.
        /// </summary>
        public GameObject SpawnPlaced(SessionData session)
        {
            if (_placedInstance != null) Destroy(_placedInstance);

            var pos = placedSpawnPoint != null ? placedSpawnPoint.position : new Vector3(2f, 0f, 0f);
            var rot = placedSpawnPoint != null ? placedSpawnPoint.rotation : Quaternion.identity;

            _placedInstance = SpawnCharacter(session.assets?.character, pos, rot, "Placed");

            // TODO: 유니티 개발자 — 모션 컴포넌트 추가
            // _placedInstance?.AddComponent<BlazePoseRetargeting>();
            // _placedInstance?.AddComponent<MixamoMotionPlayer>();

            return _placedInstance;
        }

        // ════════════════════════════════════════════════════════════
        // 내부 구현
        // ════════════════════════════════════════════════════════════

        private GameObject SpawnCharacter(
            CharacterAssetData data, Vector3 pos, Quaternion rot, string role)
        {
            if (data == null)
            {
                Debug.LogWarning($"[CharacterSpawner] CharacterAssetData 없음 ({role})");
                return null;
            }

            // Server Mode: model_url 있으면 런타임 다운로드
            if (!string.IsNullOrEmpty(data.model_url))
                return SpawnFromServer(data, pos, rot, role);

            // Mock Mode: Inspector Prefab 사용
            if (mockCharacterPrefab == null)
            {
                Debug.LogWarning($"[CharacterSpawner] mockCharacterPrefab 없음 ({role}). " +
                                 "GeneratedCharacters/mocking/character/ 에 FBX 넣고 Prefab 연결하세요.");
                return null;
            }
            var go = Instantiate(mockCharacterPrefab, pos, rot);
            go.name = $"Character_{role}_Mock";
            Debug.Log($"[CharacterSpawner] Mock 생성 ({role}): {data.npc_name}");
            return go;
        }

        private GameObject SpawnFromServer(
            CharacterAssetData data, Vector3 pos, Quaternion rot, string role)
        {
            // TODO: 유니티 개발자 — model_url(FBX) 런타임 로드 구현
            // 필요 패키지: TriLib (권장) 또는 자체 FBX 파서
            //
            // TriLib 예시:
            //   var options = AssetLoaderOptions.CreateInstance();
            //   AssetLoader.LoadModelFromUri(
            //       data.model_url, OnLoad, OnMaterials, OnProgress, OnError, null, options);
            //
            // 텍스쳐(texture_url GLB) 적용 — glTFast:
            //   var gltf = new GltfImport();
            //   await gltf.Load(data.texture_url);
            //   skinnedRenderer.material = gltf.GetMaterial(0);

            Debug.Log($"[CharacterSpawner] Server Mode FBX 로드 예정 ({role}): {data.model_url}");

            // 임시 fallback: FBX 로더 미구현 시 Mock Prefab 사용
            if (mockCharacterPrefab != null)
            {
                var go = Instantiate(mockCharacterPrefab, pos, rot);
                go.name = $"Character_{role}_ServerFallback";
                return go;
            }
            return null;
        }
    }
}
