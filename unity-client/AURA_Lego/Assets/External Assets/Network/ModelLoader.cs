using System.Collections;
using System.IO;
using GLTFast;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// BackendClient의 OnModelReady 이벤트를 받아
/// HTTP로 GLB를 다운로드한 뒤 씬에 배치한다.
///
/// 흐름:
///   WebSocket model_ready 수신
///   → GET http://backend:8000/unity/models/{filename}
///   → Application.persistentDataPath 에 저장
///   → GLTFast로 로드 → 씬 배치
/// </summary>
public class ModelLoader : MonoBehaviour
{
    [Header("Backend")]
    [SerializeField] private string _backendHost = "localhost";
    [SerializeField] private int    _backendPort = 8000;

    [Header("Spawn")]
    [SerializeField] private Vector3 _spawnPosition = Vector3.zero;
    [SerializeField] private Vector3 _spawnRotation = Vector3.zero;
    [SerializeField] private float   _spawnScale    = 1f;

    private GameObject _current;

    private string ModelCacheDir => Path.Combine(Application.persistentDataPath, "models");

    private void Awake()
    {
        Directory.CreateDirectory(ModelCacheDir);
    }

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
        string url      = $"http://{_backendHost}:{_backendPort}/unity/models/{filename}";
        string savePath = Path.Combine(ModelCacheDir, filename);
        StartCoroutine(DownloadAndLoad(url, savePath));
    }

    private IEnumerator DownloadAndLoad(string url, string savePath)
    {
        Debug.Log($"[ModelLoader] 다운로드: {url}");

        using var req = UnityWebRequest.Get(url);
        req.downloadHandler = new DownloadHandlerFile(savePath);
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"[ModelLoader] 다운로드 실패: {req.error}");
            yield break;
        }

        Debug.Log($"[ModelLoader] 저장 완료: {savePath}");
        yield return LoadGlb(savePath);
    }

    private IEnumerator LoadGlb(string path)
    {
        if (_current != null)
            Destroy(_current);

        var gltf = new GltfImport();
        var task = gltf.LoadFile(path);
        yield return new WaitUntil(() => task.IsCompleted);

        if (!task.Result)
        {
            Debug.LogError($"[ModelLoader] GLB 로드 실패: {path}");
            yield break;
        }

        _current = new GameObject("LegoModel");
        _current.transform.position    = _spawnPosition;
        _current.transform.eulerAngles = _spawnRotation;
        _current.transform.localScale  = Vector3.one * _spawnScale;

        var instantiate = gltf.InstantiateMainSceneAsync(_current.transform);
        yield return new WaitUntil(() => instantiate.IsCompleted);

        Debug.Log($"[ModelLoader] 배치 완료: {Path.GetFileName(path)}");
    }
}
