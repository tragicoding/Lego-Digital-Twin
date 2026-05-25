using UnityEngine;

public class FastSkyDayNight : MonoBehaviour
{
    [Header("시간 설정")]
    [Range(0, 24)] public float timeOfDay = 12f; // 시작 시간
    public float timeMultiplier = 1f; // 시간 배속

    [Header("URP 조명 설정")]
    public Light sun; // 메인 Directional Light
    private float sunInitialIntensity; // 태양의 초기 밝기

    void Start()
    {
        if (sun != null)
        {
            sunInitialIntensity = sun.intensity;
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
        timeOfDay += Time.deltaTime * timeMultiplier;
        if (timeOfDay >= 24f)
        {
            timeOfDay %= 24f;
        }
    }

    private void UpdateSunRotation()
    {
        if (sun == null) return;

        // 6시를 일출(0도), 18시를 일몰(180도)로 계산
        float sunRotation = (timeOfDay - 6f) / 24f * 360f;
        
        // FastSky 구름의 명암이 예쁘게 보이도록 Y축이나 Z축 각도를 약간 틀어주는 것도 좋습니다.
        sun.transform.rotation = Quaternion.Euler(sunRotation, -30f, 0f);
    }

    private void UpdateURPLighting()
    {
        if (sun == null) return;

        // 태양 벡터와 아래(Down) 벡터의 내적을 통해 태양의 높이 계산
        float dotProduct = Vector3.Dot(sun.transform.forward, Vector3.down);

        if (dotProduct > 0)
        {
            // 낮 시간대: 태양의 높이에 따라 밝기가 서서히 변함
            sun.intensity = Mathf.Lerp(0, sunInitialIntensity, dotProduct);
            sun.shadows = LightShadows.Soft; 
        }
        else
        {
            // 밤 시간대: 밝기와 그림자만 0으로 (성능 최적화)
            sun.intensity = 0f;
            sun.shadows = LightShadows.None; 
        }

        // FastSky 스카이박스 셰이더가 태양 디스크를 땅 밑으로 부드럽게 넘기려면 
        // 낮이든 밤이든 이 방향 값이 매 프레임 전달되어야 합니다.
        Shader.SetGlobalVector("_SunDir", -sun.transform.forward);
    }
}