using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Object
{
    /// <summary>
    /// 세션 데이터 기반 오브제 생성.
    ///
    /// Mock Mode  : objectPrefab을 Instantiate한다.
    /// Server Mode: model_url에서 GLB를 다운로드해 로드한다. (DownloadManager 연동 예정)
    /// </summary>
    public class GeneratedObjectSpawner : MonoBehaviour
    {
        [Header("Mock Mode용 기본 Prefab")]
        public GameObject objectPrefab;

        [Header("배치 위치")]
        public Transform spawnPoint;

        private GameObject _spawnedObject;

        /// <summary>ObjectAssetData로 오브제를 생성한다.</summary>
        public GameObject Spawn(ObjectAssetData data)
        {
            if (_spawnedObject != null)
                Destroy(_spawnedObject);

            Vector3 pos = spawnPoint != null ? spawnPoint.position : Vector3.zero;
            Quaternion rot = spawnPoint != null ? spawnPoint.rotation : Quaternion.identity;

            if (string.IsNullOrEmpty(data.model_url))
            {
                // Mock Mode: prefab 사용
                if (objectPrefab == null)
                {
                    Debug.LogWarning("[GeneratedObjectSpawner] objectPrefab이 없습니다.");
                    return null;
                }
                _spawnedObject = Instantiate(objectPrefab, pos, rot);
            }
            else
            {
                // Server Mode: model_url GLB 로드 (TODO: DownloadManager 연동)
                Debug.Log($"[GeneratedObjectSpawner] GLB 로드 예정: {data.model_url}");
                if (objectPrefab != null)
                    _spawnedObject = Instantiate(objectPrefab, pos, rot);
            }

            Debug.Log($"[GeneratedObjectSpawner] 오브제 생성: {data.object_name}");
            return _spawnedObject;
        }
    }
}
