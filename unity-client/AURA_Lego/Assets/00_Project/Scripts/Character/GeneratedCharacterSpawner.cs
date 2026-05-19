using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 세션 데이터 기반 캐릭터 생성.
    ///
    /// Mock Mode  : characterPrefab을 Instantiate한다.
    /// Server Mode: model_url에서 GLB를 다운로드해 로드한다. (DownloadManager 연동 예정)
    /// </summary>
    public class GeneratedCharacterSpawner : MonoBehaviour
    {
        [Header("Mock Mode용 기본 Prefab")]
        public GameObject characterPrefab;

        [Header("Spawn 위치")]
        public Transform spawnPoint;

        private GameObject _spawnedCharacter;

        /// <summary>CharacterAssetData로 캐릭터를 생성한다.</summary>
        public GuideNPCController Spawn(CharacterAssetData data, string bubbleText = "")
        {
            if (_spawnedCharacter != null)
                Destroy(_spawnedCharacter);

            Vector3 pos = spawnPoint != null ? spawnPoint.position : Vector3.zero;
            Quaternion rot = spawnPoint != null ? spawnPoint.rotation : Quaternion.identity;

            if (string.IsNullOrEmpty(data.model_url))
            {
                // Mock Mode: prefab 사용
                if (characterPrefab == null)
                {
                    Debug.LogWarning("[GeneratedCharacterSpawner] characterPrefab이 없습니다.");
                    return null;
                }
                _spawnedCharacter = Instantiate(characterPrefab, pos, rot);
            }
            else
            {
                // Server Mode: model_url GLB 로드 (TODO: DownloadManager 연동)
                Debug.Log($"[GeneratedCharacterSpawner] GLB 로드 예정: {data.model_url}");
                // 임시로 prefab 사용
                if (characterPrefab != null)
                    _spawnedCharacter = Instantiate(characterPrefab, pos, rot);
            }

            if (_spawnedCharacter == null) return null;

            var npc = _spawnedCharacter.GetComponent<GuideNPCController>();
            if (npc != null) npc.Initialize(data, bubbleText);

            Debug.Log($"[GeneratedCharacterSpawner] 캐릭터 생성: {data.npc_name}");
            return npc;
        }
    }
}
