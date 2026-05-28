using System.Collections;
using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Object
{
    /// <summary>
    /// 세션 데이터로 오브제를 스폰합니다.
    ///
    /// ── Mock Mode ──────────────────────────────────────────────────
    /// objectPrefab 을 spawnPoint 위치에 Instantiate합니다.
    /// Inspector 에서 objectPrefab 과 spawnPoint 를 연결하세요.
    ///
    /// ── Server Mode ────────────────────────────────────────────────
    /// data.model_url 이 있으면 glTFast 로 GLB 를 런타임 로드합니다.
    /// objectPrefab 없이도 자동으로 스폰됩니다.
    /// 로드 실패 시 objectPrefab 으로 fallback합니다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] objectPrefab 연결 (Mock 모드 및 Server 실패 시 fallback)
    ///   [ ] spawnPoint 위치 지정
    /// </summary>
    public class GeneratedObjectSpawner : MonoBehaviour
    {
        [Header("Mock Mode — 오브제 Prefab 연결")]
        [Tooltip("Mock 모드 및 Server GLB 로드 실패 시 사용할 Prefab")]
        public GameObject objectPrefab;

        [Header("스폰 위치")]
        public Transform spawnPoint;

        [Header("스폰 설정")]
        [Tooltip("스폰 시 적용할 Scale (x·y·z 동일)")]
        public float spawnScale = 1f;

        private GameObject _spawnedObject;

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// ObjectAssetData 로 오브제를 스폰합니다.
        /// model_url 없음 → Mock Mode (즉시 스폰)
        /// model_url 있음 → Server Mode (glTFast 비동기 로드, 완료 후 스폰)
        /// </summary>
        public void Spawn(ObjectAssetData data)
        {
            if (data == null)
            {
                Debug.LogWarning("[GeneratedObjectSpawner] ObjectAssetData 없음");
                return;
            }

            if (_spawnedObject != null)
                Destroy(_spawnedObject);

            if (string.IsNullOrEmpty(data.model_url))
                SpawnMock(data);
            else
                StartCoroutine(SpawnFromServerCoroutine(data));
        }

        // ════════════════════════════════════════════════════════════
        // Mock Mode
        // ════════════════════════════════════════════════════════════

        private void SpawnMock(ObjectAssetData data)
        {
            if (objectPrefab == null)
            {
                Debug.LogWarning("[GeneratedObjectSpawner] objectPrefab 없음. " +
                                 "Inspector 에서 objectPrefab 을 연결하세요.");
                return;
            }

            var pos = spawnPoint != null ? spawnPoint.position : Vector3.zero;
            var rot = spawnPoint != null ? spawnPoint.rotation : Quaternion.identity;

            _spawnedObject = Instantiate(objectPrefab, pos, rot);
            _spawnedObject.name = $"Object_{data.object_name}";
            _spawnedObject.transform.localScale = Vector3.one * spawnScale;
            Debug.Log($"[GeneratedObjectSpawner] Mock 스폰: {data.object_name} / scale: {spawnScale}");
        }

        // ════════════════════════════════════════════════════════════
        // Server Mode — glTFast GLB 비동기 로드
        // ════════════════════════════════════════════════════════════

        private IEnumerator SpawnFromServerCoroutine(ObjectAssetData data)
        {
            Debug.Log($"[GeneratedObjectSpawner] GLB 로드 시작: {data.model_url}");

            var task = SpawnFromServerAsync(data);
            yield return new WaitUntil(() => task.IsCompleted);

            if (task.IsFaulted)
                Debug.LogError($"[GeneratedObjectSpawner] GLB 로드 실패: {task.Exception?.InnerException?.Message}");
        }

        private async System.Threading.Tasks.Task SpawnFromServerAsync(ObjectAssetData data)
        {
            var pos = spawnPoint != null ? spawnPoint.position : Vector3.zero;
            var rot = spawnPoint != null ? spawnPoint.rotation : Quaternion.identity;

            var gltf = new GLTFast.GltfImport();
            bool ok = await gltf.Load(data.model_url);

            if (!ok)
            {
                Debug.LogWarning($"[GeneratedObjectSpawner] GLB 로드 실패: {data.model_url}\nMock fallback 사용.");
                SpawnMock(data);
                return;
            }

            var root = new GameObject($"Object_{data.object_name}");
            root.transform.SetPositionAndRotation(pos, rot);
            root.transform.localScale = Vector3.one * spawnScale;

            await gltf.InstantiateMainSceneAsync(root.transform);
            _spawnedObject = root;
            Debug.Log($"[GeneratedObjectSpawner] Server GLB 스폰 완료: {data.object_name}");
        }
    }
}
