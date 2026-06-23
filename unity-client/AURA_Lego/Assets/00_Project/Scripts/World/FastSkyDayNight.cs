using UnityEngine;
using LegoTwin.Managers;

public class FastSkyDayNight : MonoBehaviour
{
    [Header("시간 설정")]
    [Range(0, 24)] public float timeOfDay = 12f; // 시작 시간
    public float timeMultiplier = 1f; // 시간 배속

    [Tooltip("켜면 가이드 시작(GameFlowManager.GuideStarted) 전까지 시간을 멈춘다. 대기화면 동안 시작 시각으로 고정.\n끄면 기존처럼 Start부터 바로 흐른다(예: 하늘 테스트 씬).")]
    public bool waitForGuideStart = true;

    [Header("URP 조명 설정")]
    public Light sun; // 메인 Directional Light
    private float sunInitialIntensity;

    // FastSky 머티리얼 원본값 캐시 (낮/밤 전환 시 복원용)
    private Material _skyMat;
    private float    _originalSunIntensity;

    void Start()
    {
        if (sun != null)
            sunInitialIntensity = sun.intensity;

        // RenderSettings.skybox 가 인스턴스가 아니면 직접 수정이 에셋에 반영되므로
        // 런타임 인스턴스를 만들어서 사용한다
        if (RenderSettings.skybox != null)
        {
            _skyMat = new Material(RenderSettings.skybox);
            RenderSettings.skybox = _skyMat;
            _originalSunIntensity = _skyMat.HasFloat("_SunIntensity")
                ? _skyMat.GetFloat("_SunIntensity") : 100f;
        }
    }

    void Update()
    {
        UpdateTime();
        UpdateSunRotation();
        UpdateURPLighting();
    }

    private void UpdateTime()
    {
        // 가이드 시작 전(대기화면)에는 시간을 멈춰 시작 시각을 유지한다.
        // 태양 회전·조명은 계속 갱신되므로 하늘은 timeOfDay(시작값)로 정상 표시된다.
        if (waitForGuideStart && !GameFlowManager.GuideStarted)
            return;

        timeOfDay += Time.deltaTime * timeMultiplier;
        if (timeOfDay >= 24f)
            timeOfDay %= 24f;
    }

    private void UpdateSunRotation()
    {
        if (sun == null) return;

        float sunRotation = (timeOfDay - 6f) / 24f * 360f;
        sun.transform.rotation = Quaternion.Euler(sunRotation, -30f, 0f);

        // 밤이면 태양을 정 아래(270°)로 강제 이동 — Light 방향 기반 스카이에서도 디스크 사라짐
        if (Vector3.Dot(sun.transform.forward, Vector3.down) <= 0f)
            sun.transform.rotation = Quaternion.Euler(270f, -30f, 0f);
    }

    private void UpdateURPLighting()
    {
        if (sun == null) return;

        float dotProduct = Vector3.Dot(sun.transform.forward, Vector3.down);
        bool isDay = dotProduct > 0f;

        // ── 조명 ─────────────────────────────────────────────────────
        if (isDay)
        {
            sun.intensity = Mathf.Lerp(0, sunInitialIntensity, dotProduct);
            sun.shadows   = LightShadows.Soft;
        }
        else
        {
            sun.intensity = 0f;
            sun.shadows   = LightShadows.None;
        }

        // ── 스카이박스 태양 디스크 ───────────────────────────────────
        Vector3 sunDir = isDay ? -sun.transform.forward : Vector3.down;

        // 1) 글로벌 (일부 셰이더 대응)
        Shader.SetGlobalVector("_SunDir", sunDir);

        // 2) FastSky ShaderGraph 머티리얼 직접 설정
        //    ShaderGraph 프로퍼티는 material-level 이므로 SetGlobalVector 로는 반영 안 됨
        if (_skyMat != null)
        {
            if (_skyMat.HasProperty("_SunDir"))
                _skyMat.SetVector("_SunDir", sunDir);

            // 밤: 태양 디스크 강도를 0으로 → 디스크 완전히 소멸
            if (_skyMat.HasProperty("_SunIntensity"))
                _skyMat.SetFloat("_SunIntensity", isDay ? _originalSunIntensity : 0f);
        }
    }
}
