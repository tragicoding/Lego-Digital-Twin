using System.IO;
using System.Threading.Tasks;
using GLTFast;
using UnityEngine;

/// <summary>
/// BackendClient의 OnModelReady 이벤트를 받아
/// External Assets/capstone1/3d_model/ 에서 GLB를 로드하고 씬에 배치한다.
/// </summary>
public class ModelLoader : MonoBehaviour
{
    [Header("Spawn")]
    [SerializeField] private Vector3 _spawnPosition = Vector3.zero;
    [SerializeField] private Vector3 _spawnRotation = Vector3.zero;
    [SerializeField] private float   _spawnScale    = 1f;

    private static readonly string _modelDir = Path.Combine(
        Application.dataPath,
        "External Assets", "capstone1", "3d_model"
    );

    private GameObject _current;

    private void OnEnable()
    {
        if (BackendClient.Instance != null)
            BackendClient.Instance.OnModelReady += HandleModelReady;
    }

    private void OnDisable()
    {
        if (BackendClient.Instance != null)
            BackendClient.Instance.OnModelReady -= HandleModelReady;
    }

    private void HandleModelReady(string filename)
    {
        string path = Path.Combine(_modelDir, filename);
        if (!File.Exists(path))
        {
            Debug.LogWarning($"[ModelLoader] 파일 없음: {path}");
            return;
        }
        _ = LoadGlbAsync(path);
    }

    private async Task LoadGlbAsync(string path)
    {
        // 이전 모델 제거
        if (_current != null)
            Destroy(_current);

        var gltf = new GltfImport();
        bool ok = await gltf.LoadFile(path);
        if (!ok)
        {
            Debug.LogError($"[ModelLoader] GLB 로드 실패: {path}");
            return;
        }

        _current = new GameObject("LegoModel");
        _current.transform.position    = _spawnPosition;
        _current.transform.eulerAngles = _spawnRotation;
        _current.transform.localScale  = Vector3.one * _spawnScale;

        await gltf.InstantiateMainSceneAsync(_current.transform);
        Debug.Log($"[ModelLoader] 로드 완료: {Path.GetFileName(path)}");
    }
}
