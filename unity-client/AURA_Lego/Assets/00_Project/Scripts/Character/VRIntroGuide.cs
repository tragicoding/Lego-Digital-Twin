using System.Collections;
using UnityEngine;

public class VRIntroGuide : MonoBehaviour
{
    [Header("이동 및 대기 설정")]
    [Tooltip("출발 전 출발 위치에서 대기하는 시간 (초)")]
    public float waitTimeBeforeStart = 5.0f;
    
    [Tooltip("가이드가 시작될 위치 (비워두면 현재 XR Origin의 위치를 시작점으로 사용합니다)")]
    public Transform startPoint;
    
    [Tooltip("가이드가 도착할 최종 목표 위치")]
    public Transform targetPoint;
    
    [Tooltip("이동하는 데 걸리는 시간 (초)")]
    public float guideDuration = 5.0f;

    [Header("제어할 이동 스크립트 (선택 사항)")]
    [Tooltip("대기 및 가이드 중에는 비활성화하고, 도착 후 활성화할 플레이어 이동 스크립트")]
    public MonoBehaviour playerMovementScript;

    private void Start()
    {
        if (targetPoint == null)
        {
            Debug.LogError("목표 위치(Target Point)가 설정되지 않았습니다!");
            return;
        }

        // 씬 시작과 동시에 가이드 코루틴 실행
        StartCoroutine(GuidePlayerCoroutine());
    }

    private IEnumerator GuidePlayerCoroutine()
    {
        // 1. 코루틴이 시작되자마자 플레이어의 수동 조작 방지
        if (playerMovementScript != null)
        {
            playerMovementScript.enabled = false;
        }

        // 시작 지점이 지정되어 있다면 해당 위치로 즉시 이동
        if (startPoint != null)
        {
            transform.position = startPoint.position;
            transform.rotation = startPoint.rotation;
        }

        // 2. 지정된 시간(waitTimeBeforeStart)만큼 출발 위치에서 대기
        yield return new WaitForSeconds(waitTimeBeforeStart);

        // 대기가 끝난 시점의 위치와 회전값을 이동 시작점으로 저장
        Vector3 initialPosition = transform.position;
        Quaternion initialRotation = transform.rotation;

        float elapsedTime = 0f;

        // 3. 지정된 시간(guideDuration) 동안 부드럽게 이동 (Lerp 사용)
        while (elapsedTime < guideDuration)
        {
            // 시간에 따른 위치 및 회전 보간
            transform.position = Vector3.Lerp(initialPosition, targetPoint.position, elapsedTime / guideDuration);
            transform.rotation = Quaternion.Lerp(initialRotation, targetPoint.rotation, elapsedTime / guideDuration);
            
            elapsedTime += Time.deltaTime;
            yield return null; // 다음 프레임까지 대기
        }

        // 4. 목표 지점에 오차 없이 정확히 안착
        transform.position = targetPoint.position;
        transform.rotation = targetPoint.rotation;

        // 5. 도착 후 플레이어의 수동 조작 권한 복구
        if (playerMovementScript != null)
        {
            playerMovementScript.enabled = true;
        }

        Debug.Log("인트로 대기 및 가이드 이동이 완료되었습니다.");
    }
}