using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 캐릭터 스포너.
    /// 같은 캐릭터 데이터를 두 역할로 생성한다:
    ///   1. Guide  — 가이드 NPC (걸어다니며 안내)
    ///   2. Placed — 광장 배치 캐릭터 (오브제 옆, 모션 수행)
    ///
    /// ── 컴포넌트 자동 주입 ──────────────────────────────────────────
    /// Mock Prefab / 서버 FBX 어느 쪽이든 스폰 후 SetupAsGuide / SetupAsPlaced 를
    /// 호출하면 필요한 컴포넌트를 자동으로 추가·연결한다.
    ///   - 이미 붙어있으면 재사용 (덮어쓰지 않음)
    ///   - 없으면 AddComponent 로 추가
    ///   - GuideNPCController.Animation 필드 자동 연결
    ///   - PlacedCharacterController.motionLibrary 자동 연결
    ///
    /// ── Mock → Server 전환 ───────────────────────────────────────────
    /// LoadFromServer() 안 TODO 완성 후 SetupAsGuide / SetupAsPlaced 를
    /// 콜백 안에서 호출하면 나머지 코드 변경 없이 서버 캐릭터 자동 설정된다.
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] mockCharacterPrefab 연결 (Mock 모드)
    ///   [ ] guideSpawnPoint, placedSpawnPoint 위치 지정
    ///   [ ] motionLibrary ScriptableObject 연결
    /// </summary>
    public class GeneratedCharacterSpawner : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("Mock Mode — FBX 로 만든 Prefab 연결")]
        public GameObject mockCharacterPrefab;

        [Header("스폰 위치")]
        [Tooltip("가이드 NPC 스폰 위치 (등장 시작점)")]
        public Transform guideSpawnPoint;
        [Tooltip("가이드 NPC 가 걸어가서 멈출 위치 (인사 위치). 비워두면 스폰 위치에서 즉시 인사.")]
        public Transform   guideArrivalPoint;
        [Tooltip("광장까지 이동 경로 웨이포인트 (순서대로 연결)")]
        public Transform[] plazaPathWaypoints;
        [Tooltip("내 창작물 앞 위치 (캐릭터+오브제 확인 위치)")]
        public Transform   myCreationWaypoint;
        [Tooltip("광장 배치 캐릭터 위치 (오브제 옆)")]
        public Transform placedSpawnPoint;

        [Header("스폰 설정")]
        [Tooltip("스폰 시 적용할 Scale (x·y·z 동일)")]
        public float spawnScale = 6f;

        [Header("컴포넌트 자동 주입")]
        [Tooltip("PlacedCharacterController 에 자동 연결할 MixamoMotionLibrary ScriptableObject")]
        public MixamoMotionLibrary motionLibrary;

        [Tooltip("캐릭터 Animator 에 자동 연결할 Animator Controller (없으면 스킵)")]
        public RuntimeAnimatorController animatorController;

        // ── 런타임 인스턴스 참조 ─────────────────────────────────────
        private GameObject _guideInstance;
        private GameObject _placedInstance;

        // ════════════════════════════════════════════════════════════
        // 공개 API
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 가이드 NPC 생성 후 GuideNPCController 반환.
        /// 컴포넌트가 없는 raw FBX 오브젝트에도 자동으로 컴포넌트를 추가·연결한다.
        /// </summary>
        public GuideNPCController SpawnGuide(SessionData session)
        {
            if (_guideInstance != null) Destroy(_guideInstance);

            var pos = guideSpawnPoint != null ? guideSpawnPoint.position : Vector3.zero;
            var rot = guideSpawnPoint != null ? guideSpawnPoint.rotation : Quaternion.identity;

            var go = CreateCharacterObject(session.assets?.character, pos, rot, "Guide");
            if (go == null) return null;

            _guideInstance = go;
            return SetupAsGuide(go, session);   // ← 컴포넌트 자동 주입
        }

        /// <summary>
        /// 광장 배치 캐릭터 생성 후 PlacedCharacterController 반환.
        /// 컴포넌트가 없는 raw FBX 오브젝트에도 자동으로 컴포넌트를 추가·연결한다.
        /// </summary>
        public PlacedCharacterController SpawnPlaced(SessionData session)
        {
            if (_placedInstance != null) Destroy(_placedInstance);

            var pos = placedSpawnPoint != null ? placedSpawnPoint.position : new Vector3(2f, 0f, 0f);
            var rot = placedSpawnPoint != null ? placedSpawnPoint.rotation : Quaternion.identity;

            var go = CreateCharacterObject(session.assets?.character, pos, rot, "Placed");
            if (go == null) return null;

            _placedInstance = go;
            return SetupAsPlaced(go, session);  // ← 컴포넌트 자동 주입
        }

        // ════════════════════════════════════════════════════════════
        // 컴포넌트 자동 주입
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// Guide NPC 에 필요한 컴포넌트를 자동으로 추가·연결한다.
        /// 서버 FBX 로드 완료 콜백에서도 동일하게 호출하면 된다:
        ///   SetupAsGuide(serverLoadedGO, session)
        /// </summary>
        private GuideNPCController SetupAsGuide(GameObject go, SessionData session)
        {
            SetupAnimatorController(go);

            // 있으면 재사용, 없으면 AddComponent
            var anim = GetOrAdd<CharacterAnimationController>(go);
            var npc  = GetOrAdd<GuideNPCController>(go);

            // Animation 필드 — Prefab 에 이미 연결돼 있으면 유지, 없을 때만 주입
            if (npc.Animation == null)
                npc.Animation = anim;

            // npcName · sessionId 세팅 + anim.Initialize() 내부 호출
            npc.Initialize(session);

            // 웨이포인트 전달 — 씬에 배치된 Transform 을 NPC 에 주입
            if (guideArrivalPoint != null)
                npc.guideArrivalPoint = guideArrivalPoint;
            if (plazaPathWaypoints is { Length: > 0 })
                npc.plazaPathWaypoints = plazaPathWaypoints;
            if (myCreationWaypoint != null)
                npc.myCreationWaypoint = myCreationWaypoint;

            Debug.Log($"[CharacterSpawner] Guide 설정 완료: {session.character_npc_name}");
            return npc;
        }

        /// <summary>
        /// Placed 캐릭터에 필요한 컴포넌트를 자동으로 추가·연결한다.
        /// 서버 FBX 로드 완료 콜백에서도 동일하게 호출하면 된다:
        ///   SetupAsPlaced(serverLoadedGO, session)
        /// </summary>
        private PlacedCharacterController SetupAsPlaced(GameObject go, SessionData session)
        {
            SetupAnimatorController(go);

            var anim   = GetOrAdd<CharacterAnimationController>(go);
            var placed = GetOrAdd<PlacedCharacterController>(go);
            // PlacedCharacterController.Awake() 가 _animation 을 GetComponent 로 찾으므로
            // CharacterAnimationController 를 먼저 추가한 뒤 PlacedCharacterController 추가하면 자동 연결됨

            // motionLibrary — Inspector 연결 없을 때만 주입
            if (placed.motionLibrary == null)
                placed.motionLibrary = motionLibrary;

            // 캐릭터 이름·말풍선 세팅
            anim.Initialize(session.assets?.character, session.bubble_text);

            Debug.Log($"[CharacterSpawner] Placed 설정 완료: {session.character_npc_name}");
            return placed;
        }

        // ════════════════════════════════════════════════════════════
        // 오브젝트 생성 (Mock / Server 분기)
        // ════════════════════════════════════════════════════════════

        private GameObject CreateCharacterObject(
            CharacterAssetData data, Vector3 pos, Quaternion rot, string role)
        {
            if (data == null)
            {
                Debug.LogWarning($"[CharacterSpawner] CharacterAssetData 없음 ({role})");
                return null;
            }

            // Server Mode: model_url 있으면 런타임 로드
            if (!string.IsNullOrEmpty(data.model_url))
                return LoadFromServer(data, pos, rot, role);

            // Mock Mode: Inspector Prefab 사용
            if (mockCharacterPrefab == null)
            {
                Debug.LogWarning($"[CharacterSpawner] mockCharacterPrefab 없음 ({role}). " +
                                 "Inspector 에서 mockCharacterPrefab 을 연결하세요.");
                return null;
            }

            var go = Instantiate(mockCharacterPrefab, pos, rot);
            go.name = $"Character_{role}";
            go.transform.localScale = Vector3.one * spawnScale;
            Debug.Log($"[CharacterSpawner] Mock 생성 ({role}): {data.npc_name} / scale: {spawnScale}");
            return go;
        }

        private GameObject LoadFromServer(
            CharacterAssetData data, Vector3 pos, Quaternion rot, string role)
        {
            // TODO: TriLib 또는 glTFast 로 FBX/GLB 런타임 로드 후 SetupAsGuide/SetupAsPlaced 호출
            //
            // TriLib 예시 (Guide 기준):
            //   var options = AssetLoaderOptions.CreateInstance();
            //   AssetLoader.LoadModelFromUri(
            //       data.model_url,
            //       ctx => {
            //           var go = ctx.RootGameObject;
            //           go.transform.SetPositionAndRotation(pos, rot);
            //           go.transform.localScale = Vector3.one * spawnScale;
            //           HumanoidAvatarBuilder.Build(go);   // ← Humanoid Avatar 자동 구성
            //           _guideInstance = go;
            //           SetupAsGuide(go, session);         // ← 컴포넌트 자동 주입
            //       },
            //       null, null, null, null, options
            //   );
            //
            // 텍스처(GLB) 적용 — glTFast:
            //   var gltf = new GltfImport();
            //   await gltf.Load(data.texture_url);
            //   renderer.material = gltf.GetMaterial(0);

            Debug.Log($"[CharacterSpawner] Server Mode ({role}): {data.model_url} — TODO: TriLib 연동");

            // 임시 fallback: FBX 로더 미구현 시 Mock Prefab 사용
            if (mockCharacterPrefab == null) return null;

            var fallback = Instantiate(mockCharacterPrefab, pos, rot);
            fallback.name = $"Character_{role}_ServerFallback";
            fallback.transform.localScale = Vector3.one * spawnScale;
            return fallback;
        }

        // ════════════════════════════════════════════════════════════
        // 유틸
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// Animator Controller 가 Inspector 에 연결되어 있고 캐릭터 Animator 에
        /// 아직 Controller 가 없을 때만 자동으로 할당한다.
        /// </summary>
        private void SetupAnimatorController(GameObject go)
        {
            if (animatorController == null) return;

            var animator = go.GetComponentInChildren<Animator>();
            if (animator == null || animator.runtimeAnimatorController != null) return;

            animator.runtimeAnimatorController = animatorController;
            Debug.Log($"[CharacterSpawner] Animator Controller 자동 연결: {go.name}");
        }

        /// <summary>
        /// T 컴포넌트가 go 에 있으면 반환, 없으면 AddComponent 후 반환.
        /// Mock Prefab / 서버 FBX 모두 동일하게 동작한다.
        /// </summary>
        private static T GetOrAdd<T>(GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            return c != null ? c : go.AddComponent<T>();
        }
    }
}
