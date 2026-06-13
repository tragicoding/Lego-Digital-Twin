using UnityEngine;

namespace LegoTwin.Object
{
    /// <summary>
    /// 런타임 생성 오브젝트에 전시용 단순 물리를 적용하는 유틸.
    /// 모든 Renderer의 world bounds를 합산한 BoxCollider를 루트에 항상 재계산한다.
    /// 프리팹에 기존 Collider가 있어도 루트 BoxCollider는 덮어쓴다.
    /// (피벗이 메시 중앙에 있는 프리팹의 절반 파묻힘 버그 방지)
    /// </summary>
    public static class RuntimePhysicsUtil
    {
        public static void SetupSimplePhysics(GameObject go, float mass = 1f)
        {
            if (go == null) return;

            var rb = go.GetComponent<Rigidbody>();
            if (rb == null)
                rb = go.AddComponent<Rigidbody>();

            rb.mass = mass;
            rb.useGravity = true;
            // Continuous: 고속 낙하 중 바닥 콜라이더를 관통하는 tunneling 방지
            rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
            // FreezePositionXZ: 낙하 중 XZ 방향 미끄러짐 방지
            rb.constraints = RigidbodyConstraints.FreezeRotation
                           | RigidbodyConstraints.FreezePositionX
                           | RigidbodyConstraints.FreezePositionZ;

            // 기존 Collider 유무 관계없이 항상 실제 bounds로 루트 BoxCollider를 재계산
            ApplyBoundsBoxCollider(go);

            // 낙하 후 안착하면 kinematic으로 전환해 상시 물리 비용 제거 (전시용)
            if (go.GetComponent<PhysicsSettler>() == null)
                go.AddComponent<PhysicsSettler>();
        }

        private static void ApplyBoundsBoxCollider(GameObject go)
        {
            var renderers = go.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length == 0)
            {
                Debug.LogWarning($"[RuntimePhysicsUtil] Renderer 없음 — Collider 생성 생략: {go.name}");
                return;
            }

            bool hasBounds = false;
            Bounds worldBounds = default;
            foreach (var r in renderers)
            {
                if (r == null) continue;
                if (!hasBounds) { worldBounds = r.bounds; hasBounds = true; }
                else worldBounds.Encapsulate(r.bounds);
            }

            if (!hasBounds)
            {
                Debug.LogWarning($"[RuntimePhysicsUtil] 유효한 bounds 없음 — Collider 생성 생략: {go.name}");
                return;
            }

            // 루트에 BoxCollider를 찾거나 새로 추가 (자식 Collider는 건드리지 않음)
            var col = go.GetComponent<BoxCollider>();
            if (col == null)
                col = go.AddComponent<BoxCollider>();

            col.center = go.transform.InverseTransformPoint(worldBounds.center);
            var size   = go.transform.InverseTransformVector(worldBounds.size);
            var absSize = new Vector3(Mathf.Abs(size.x), Mathf.Abs(size.y), Mathf.Abs(size.z));

            // renderer.bounds 미초기화(GLB 비동기 로드 등)로 size ≈ 0이면 기본값 사용
            // center=(0,0,0), size=(1,1,1)은 scale=12 기준 월드 12×12×12 → 피벗 중앙·바닥 모두 안전
            if (absSize.y < 0.001f)
            {
                Debug.LogWarning($"[RuntimePhysicsUtil] bounds.y≈0 — 기본 BoxCollider(1,1,1) 사용: {go.name}");
                col.center = Vector3.zero;
                col.size   = Vector3.one;
            }
            else
            {
                col.size = absSize;
            }

            Debug.Log($"[RuntimePhysicsUtil] BoxCollider(재)계산: {go.name} center={col.center} size={col.size}");
        }
    }
}
