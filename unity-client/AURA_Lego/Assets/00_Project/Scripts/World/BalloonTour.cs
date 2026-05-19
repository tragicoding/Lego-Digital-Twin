using UnityEngine;

public class BalloonTour : MonoBehaviour
{
    [Header("웨이포인트 배열")]
    public Transform[] waypoints;

    [Header("이동 설정")]
    public float speed = 5f;
    public float rotationSpeed = 2f;
    public bool lookAtTarget = true;

    private int currentIndex = 0;

    void Update()
    {
        if (waypoints.Length == 0) return;

        Transform targetWaypoint = waypoints[currentIndex];

        // 1. 목표 웨이포인트 방향 이동
        transform.position = Vector3.MoveTowards(
            transform.position, targetWaypoint.position, speed * Time.deltaTime);

        // 2. 이동하는 방향으로 자연스럽게 회전 (위아래 회전 없음)
        if (lookAtTarget)
        {
            Vector3 direction = targetWaypoint.position - transform.position;
            direction.y = 0; // Y축(수직) 회전 고정 — 좌우 방향으로만 회전
            if (direction != Vector3.zero)
            {
                Quaternion targetRotation = Quaternion.LookRotation(direction);
                transform.rotation = Quaternion.Slerp(
                    transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
            }
        }

        // 3. 목표 웨이포인트 도달 확인
        if (Vector3.Distance(transform.position, targetWaypoint.position) < 0.1f)
        {
            currentIndex = (currentIndex + 1) % waypoints.Length;
        }
    }
}
