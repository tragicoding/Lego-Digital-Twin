using UnityEngine;

public class BalloonTour : MonoBehaviour
{
    [Header("경로 설정")]
    public Transform[] waypoints;

    [Header("비행 설정")]
    public float speed = 5f;
    public float rotationSpeed = 2f;
    public bool lookAtTarget = true;

    private int currentIndex = 0;

    void Update()
    {
        if (waypoints.Length == 0) return;

        Transform targetWaypoint = waypoints[currentIndex];

        // 1. 목표 지점을 향해 이동
        transform.position = Vector3.MoveTowards(transform.position, targetWaypoint.position, speed * Time.deltaTime);

        // 2. 이동하는 방향으로 자연스럽게 회전 (기울어짐 방지 적용)
        if (lookAtTarget)
        {
            Vector3 direction = targetWaypoint.position - transform.position;

            // ★ 핵심 포인트: 방향 벡터의 Y축(높낮이) 값을 0으로 만들어버림
            // 이렇게 하면 오직 수평 방향(X, Z)으로만 회전각을 계산합니다.
            direction.y = 0;

            if (direction != Vector3.zero)
            {
                Quaternion targetRotation = Quaternion.LookRotation(direction);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
            }
        }

        // 3. 목표 지점에 도달했는지 확인
        if (Vector3.Distance(transform.position, targetWaypoint.position) < 0.1f)
        {
            currentIndex = (currentIndex + 1) % waypoints.Length;
        }
    }
}