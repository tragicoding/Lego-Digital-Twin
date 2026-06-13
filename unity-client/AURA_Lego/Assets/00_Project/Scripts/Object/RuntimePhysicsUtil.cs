using UnityEngine;

namespace LegoTwin.Object
{
    /// <summary>
    /// 런타임 생성 전시 오브제를 낙하 없이 바닥에 즉시 안착시키는 유틸.
    /// 모든 Renderer의 world bounds를 합산해
    ///   ① 아래로 Ground 레이어를 레이캐스트해 실제 바닥 Y를 찾고 (실패 시 fallback)
    ///   ② 메시 바닥(bounds.min.y)이 그 바닥에 닿도록 수직 위치를 보정하고
    ///   ③ 같은 bounds로 루트 BoxCollider를 재계산한다.
    /// Rigidbody·중력을 쓰지 않으므로 낙하 중 바닥 관통이나 안착 도중 동결로
    /// 인한 "땅 속 파묻힘"이 발생하지 않는다. (XZ·회전·스케일은 호출 전에 적용된 상태)
    /// </summary>
    public static class RuntimePhysicsUtil
    {
        // 바닥 탐지용 Ground 레이어 마스크. 레이어가 없으면 0 → 레이캐스트 생략.
        private static readonly int GroundMask = LayerMask.GetMask("Ground");

        /// <summary>
        /// 오브제를 바닥에 즉시 안착시킨다. (물리 낙하 없음)
        /// 오브제 footprint 중심에서 아래로 Ground 레이어를 레이캐스트해 실제 바닥 Y를 찾는다.
        /// 바닥을 못 찾으면(레이어 없음·콜라이더 없음) fallbackGroundY를 사용한다.
        /// → spawnPoint의 Y가 높게 설정돼 있어도 항상 바닥에 앉는다.
        /// </summary>
        public static void PlaceOnGround(GameObject go, float fallbackGroundY)
        {
            if (go == null) return;

            if (!TryComputeWorldBounds(go, out var bounds))
            {
                Debug.LogWarning($"[RuntimePhysicsUtil] 유효한 Renderer bounds 없음 — 안착 생략: {go.name}");
                return;
            }

            float groundY = ResolveGroundY(bounds, fallbackGroundY);

            // 메시 바닥이 groundY에 닿도록 수직 보정 (낙하 없이 즉시 안착)
            float delta = groundY - bounds.min.y;
            go.transform.position += Vector3.up * delta;
            bounds.center += Vector3.up * delta;

            // 프리팹에 Rigidbody가 박혀 있어도 중력으로 떨어지지 않도록 kinematic 고정
            var rb = go.GetComponent<Rigidbody>();
            if (rb != null) rb.isKinematic = true;

            ApplyBoundsBoxCollider(go, bounds);
        }

        // footprint 중심에서 위→아래로 Ground 레이어 레이캐스트. 못 맞히면 fallback 반환.
        private static float ResolveGroundY(Bounds bounds, float fallbackGroundY)
        {
            if (GroundMask == 0) return fallbackGroundY;

            var origin = new Vector3(bounds.center.x, bounds.center.y + 500f, bounds.center.z);
            if (Physics.Raycast(origin, Vector3.down, out var hit, 2000f,
                                GroundMask, QueryTriggerInteraction.Ignore))
                return hit.point.y;

            return fallbackGroundY;
        }

        private static bool TryComputeWorldBounds(GameObject go, out Bounds worldBounds)
        {
            worldBounds = default;
            var renderers = go.GetComponentsInChildren<Renderer>(true);
            bool has = false;
            foreach (var r in renderers)
            {
                if (r == null) continue;
                if (!has) { worldBounds = r.bounds; has = true; }
                else worldBounds.Encapsulate(r.bounds);
            }
            return has;
        }

        // 루트에 BoxCollider를 찾거나 추가해 실제 world bounds로 (재)계산한다.
        // (씬에서 콜라이더 크기가 메시와 정확히 일치함을 확인한 기존 방식 유지)
        private static void ApplyBoundsBoxCollider(GameObject go, Bounds worldBounds)
        {
            var col = go.GetComponent<BoxCollider>();
            if (col == null)
                col = go.AddComponent<BoxCollider>();

            col.center = go.transform.InverseTransformPoint(worldBounds.center);
            var size   = go.transform.InverseTransformVector(worldBounds.size);
            col.size   = new Vector3(Mathf.Abs(size.x), Mathf.Abs(size.y), Mathf.Abs(size.z));
        }
    }
}
