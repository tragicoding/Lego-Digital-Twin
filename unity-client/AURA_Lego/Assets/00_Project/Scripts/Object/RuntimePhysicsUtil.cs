using UnityEngine;

namespace LegoTwin.Object
{
    /// <summary>
    /// 런타임 생성 오브젝트에 전시용 단순 물리를 적용하는 유틸.
    /// 고폴리 메시 전체에 convex MeshCollider를 붙이지 않고,
    /// 렌더러 bounds 기반 BoxCollider 1개로 프록시 충돌체를 만든다.
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
            rb.constraints = RigidbodyConstraints.FreezeRotation;

            if (go.GetComponentInChildren<Collider>() != null) return;

            AddBoundsBoxCollider(go);
        }

        private static void AddBoundsBoxCollider(GameObject go)
        {
            var renderers = go.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length == 0)
            {
                Debug.LogWarning($"[RuntimePhysicsUtil] Renderer 없음 — Collider 생성 생략: {go.name}");
                return;
            }

            bool hasBounds = false;
            Bounds worldBounds = default;
            foreach (var renderer in renderers)
            {
                if (renderer == null) continue;

                if (!hasBounds)
                {
                    worldBounds = renderer.bounds;
                    hasBounds = true;
                }
                else
                {
                    worldBounds.Encapsulate(renderer.bounds);
                }
            }

            if (!hasBounds)
            {
                Debug.LogWarning($"[RuntimePhysicsUtil] 유효한 bounds 없음 — Collider 생성 생략: {go.name}");
                return;
            }

            var collider = go.GetComponent<BoxCollider>();
            if (collider == null)
                collider = go.AddComponent<BoxCollider>();

            var center = go.transform.InverseTransformPoint(worldBounds.center);
            var size = go.transform.InverseTransformVector(worldBounds.size);

            collider.center = center;
            collider.size = new Vector3(
                Mathf.Abs(size.x),
                Mathf.Abs(size.y),
                Mathf.Abs(size.z));

            Debug.Log($"[RuntimePhysicsUtil] BoxCollider 프록시 생성: {go.name} size={collider.size}");
        }
    }
}
