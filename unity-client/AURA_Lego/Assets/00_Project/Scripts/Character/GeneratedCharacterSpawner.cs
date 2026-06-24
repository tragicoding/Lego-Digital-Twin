using System;
using UnityEngine;
using LegoTwin.Core;
using LegoTwin.Data;
using LegoTwin.UI;

namespace LegoTwin.Character
{
    /// <summary>
    /// 캐릭터 스포너.
    /// 같은 캐릭터 데이터를 두 역할로 생성한다:
    ///   1. Guide  — 가이드 NPC (걸어다니며 안내)
    ///   2. Placed — 광장 배치 캐릭터 (오브제 옆, 모션 수행)
    ///
    /// ── 비동기 스폰 (Mock = 동기, Server = 비동기) ────────────────────
    /// SpawnGuide / SpawnPlaced 는 콜백으로 결과를 돌려준다.
    ///   - Mock   : 프리팹을 즉시 Instantiate → 콜백을 같은 프레임에 호출 (기존과 동일)
    ///   - Server : TriLib 으로 FBX 를 비동기 로드 → 완료 시 콜백 호출
    /// 두 경로 모두 로드된 GameObject 가 준비된 뒤 SetupAsGuide / SetupAsPlaced 로
    /// 합류하므로, 컴포넌트 자동 주입·초기화 로직이 한 곳에만 존재한다.
    ///
    /// ── 컴포넌트 자동 주입 ──────────────────────────────────────────
    ///   - 이미 붙어있으면 재사용, 없으면 AddComponent
    ///   - GuideNPCController.Animation / PlacedCharacterController.motionLibrary 자동 연결
    ///   - texture_url(PBR GLB) 이 있으면 CharacterAnimationController 가 머티리얼 적용
    ///
    /// 유니티 개발자 체크리스트:
    ///   [ ] mockCharacterPrefab 연결 (Mock 모드 / Server 로드 실패 시 폴백)
    ///   [ ] guideSpawnPoint, placedSpawnPoint 위치 지정
    ///   [ ] animatorController, motionLibrary 연결
    /// </summary>
    public class GeneratedCharacterSpawner : MonoBehaviour
    {
        // ── Inspector 연결 ───────────────────────────────────────────

        [Header("Mock Mode — FBX 로 만든 Prefab 연결")]
        public GameObject mockCharacterPrefab;

        [Header("Server Mode — 사전 제작 캐릭터 Resources 경로")]
        [Tooltip("Assets/**/Resources 아래 경로. 예: Resources/PrebuiltCharacters/character_1.fbx")]
        public string prebuiltCharacterResourcePrefix = "PrebuiltCharacters/character_";

        [Header("스폰 위치")]
        [Tooltip("가이드 NPC 스폰 위치 (등장 시작점)")]
        public Transform guideSpawnPoint;
        [Tooltip("가이드 NPC 가 걸어가서 멈출 위치 (인사 위치). 비워두면 스폰 위치에서 즉시 인사.")]
        public Transform   guideArrivalPoint;
        [Tooltip("내 창작물 앞 위치 — NPC 순간이동 대상")]
        public Transform   myCreationWaypoint;
        [Tooltip("창작물 확인 시 플레이어가 순간이동할 위치. 비워두면 myCreationWaypoint 위치 사용.")]
        public Transform   playerTeleportPoint;
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
        // 공개 API — 콜백 기반 (Mock 동기 / Server 비동기 공통)
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// 가이드 NPC 를 생성하고 준비되면 <paramref name="onReady"/> 로 GuideNPCController 를 돌려준다.
        /// 실패 시 onReady(null).
        /// </summary>
        public void SpawnGuide(SessionData session, Action<GuideNPCController> onReady)
        {
            if (_guideInstance != null) Destroy(_guideInstance);

            var pos = guideSpawnPoint != null ? guideSpawnPoint.position : Vector3.zero;
            var rot = guideSpawnPoint != null ? guideSpawnPoint.rotation : Quaternion.identity;

            CreateCharacterObject(session.assets?.character, pos, rot, "Guide", go =>
            {
                if (go == null) { onReady?.Invoke(null); return; }
                _guideInstance = go;
                onReady?.Invoke(SetupAsGuide(go, session));
            });
        }

        /// <summary>
        /// 광장 배치 캐릭터를 생성하고 준비되면 <paramref name="onReady"/> 로 PlacedCharacterController 를 돌려준다.
        /// 실패 시 onReady(null).
        /// </summary>
        public void SpawnPlaced(SessionData session, Action<PlacedCharacterController> onReady)
        {
            if (_placedInstance != null) Destroy(_placedInstance);

            var pos = placedSpawnPoint != null ? placedSpawnPoint.position : new Vector3(2f, 0f, 0f);
            var rot = placedSpawnPoint != null ? placedSpawnPoint.rotation : Quaternion.identity;

            CreateCharacterObject(session.assets?.character, pos, rot, "Placed", go =>
            {
                if (go == null) { onReady?.Invoke(null); return; }
                _placedInstance = go;
                onReady?.Invoke(SetupAsPlaced(go, session));
            });
        }

        // ════════════════════════════════════════════════════════════
        // 컴포넌트 자동 주입 (Mock·Server 공통 합류 지점)
        // ════════════════════════════════════════════════════════════

        private GuideNPCController SetupAsGuide(GameObject go, SessionData session)
        {
            SetupAnimatorController(go);

            var anim = GetOrAdd<CharacterAnimationController>(go);
            var npc  = GetOrAdd<GuideNPCController>(go);

            if (npc.Animation == null)
                npc.Animation = anim;

            // npcName·말풍선 + texture_url(PBR GLB) 머티리얼 적용 (Server 모드)
            anim.Initialize(session.assets?.character, session.bubble_text);
            npc.Initialize(session);

            // 가이드 등장 걷기 클립을 모션 라이브러리(MotionType.Walk)에서 가져와 적용.
            // Animator 의 Walk .anim 직접 참조에 의존하지 않게 해 Walk 변경 시 누락을 방지한다.
            anim.ApplyWalkFromLibrary(motionLibrary);

            if (guideArrivalPoint != null)
                npc.guideArrivalPoint = guideArrivalPoint;
            if (myCreationWaypoint != null)
                npc.myCreationWaypoint = myCreationWaypoint;
            if (playerTeleportPoint != null)
                npc.playerTeleportPoint = playerTeleportPoint;

            RuntimeOptimizer.Optimize(go);

            // 가이드 NPC 는 이름표를 붙이지 않는다(사용자 요청).

            return npc;
        }

        private PlacedCharacterController SetupAsPlaced(GameObject go, SessionData session)
        {
            SetupAnimatorController(go);

            var anim   = GetOrAdd<CharacterAnimationController>(go);
            var placed = GetOrAdd<PlacedCharacterController>(go);

            if (placed.motionLibrary == null)
                placed.motionLibrary = motionLibrary;

            anim.Initialize(session.assets?.character, session.bubble_text);

            RuntimeOptimizer.Optimize(go);

            // 캐릭터 머리 위 이름표(있을 때만, 캐릭터용 축소 크기) — Mock·Server 공통
            NameTagSpawner.Instance?.AttachCharacter(go, session.character_npc_name);

            return placed;
        }

        // ════════════════════════════════════════════════════════════
        // 오브젝트 생성 (Mock / Server 분기) — 콜백으로 GameObject 반환
        // ════════════════════════════════════════════════════════════

        /// <summary>
        /// CharacterAssetData 로 캐릭터 GameObject 를 만든다.
        ///   model_url 있음 → Server (TriLib FBX 비동기 로드)
        ///   model_url 없음 → Mock  (프리팹 즉시 Instantiate)
        /// 준비되면 onCreated(go), 실패 시 onCreated(null).
        /// </summary>
        private void CreateCharacterObject(
            CharacterAssetData data, Vector3 pos, Quaternion rot, string role,
            Action<GameObject> onCreated)
        {
            if (data == null)
            {
                Debug.LogWarning($"[CharacterSpawner] CharacterAssetData 없음 ({role})");
                onCreated?.Invoke(null);
                return;
            }

            if (data.character_number > 0)
            {
                var prebuilt = InstantiatePrebuilt(data.character_number, pos, rot, role);
                if (prebuilt != null)
                {
                    onCreated?.Invoke(prebuilt);
                    return;
                }
            }

            if (!string.IsNullOrEmpty(data.model_url))
                LoadFromServer(data, pos, rot, role, onCreated);
            else
                onCreated?.Invoke(InstantiateMock(pos, rot, role));
        }

        private GameObject InstantiatePrebuilt(int characterNumber, Vector3 pos, Quaternion rot, string role)
        {
            var prefab = LoadPrebuiltPrefab(characterNumber, out var resourcePath);
            if (prefab == null)
            {
                Debug.LogWarning($"[CharacterSpawner] 사전 제작 캐릭터 없음: Resources/{resourcePath}");
                return null;
            }

            var go = Instantiate(prefab, pos, rot);
            go.name = $"Character_{role}_{characterNumber}";
            go.transform.localScale = Vector3.one * spawnScale;
            return go;
        }

        private GameObject LoadPrebuiltPrefab(int characterNumber, out string resourcePath)
        {
            resourcePath = $"{prebuiltCharacterResourcePrefix}{characterNumber}";
            var prefab = Resources.Load<GameObject>(resourcePath);
            if (prefab != null) return prefab;

            const string lowerPrefix = "PrebuiltCharacters/character_";
            const string upperPrefix = "PrebuiltCharacters/Character_";

            var lowerPath = $"{lowerPrefix}{characterNumber}";
            if (!string.Equals(lowerPath, resourcePath, StringComparison.Ordinal))
            {
                prefab = Resources.Load<GameObject>(lowerPath);
                if (prefab != null)
                {
                    resourcePath = lowerPath;
                    return prefab;
                }
            }

            var upperPath = $"{upperPrefix}{characterNumber}";
            if (!string.Equals(upperPath, resourcePath, StringComparison.Ordinal) &&
                !string.Equals(upperPath, lowerPath, StringComparison.Ordinal))
            {
                prefab = Resources.Load<GameObject>(upperPath);
                if (prefab != null)
                {
                    resourcePath = upperPath;
                    return prefab;
                }
            }

            resourcePath = $"{resourcePath} (fallback tried: {lowerPath}, {upperPath})";
            return null;
        }

        private GameObject InstantiateMock(Vector3 pos, Quaternion rot, string role)
        {
            if (mockCharacterPrefab == null)
            {
                Debug.LogWarning($"[CharacterSpawner] mockCharacterPrefab 없음 ({role}). " +
                                 "Inspector 에서 mockCharacterPrefab 을 연결하세요.");
                return null;
            }

            var go = Instantiate(mockCharacterPrefab, pos, rot);
            go.name = $"Character_{role}";
            go.transform.localScale = Vector3.one * spawnScale;
            return go;
        }

        /// <summary>
        /// Server 모드 — TriLib 으로 리깅 FBX 를 런타임 로드한다.
        /// 빈 컨테이너를 먼저 만들어 위치·스케일을 잡고, 그 아래에 모델을 임포트한다.
        /// 로드 실패 시 mockCharacterPrefab 으로 폴백한다.
        /// </summary>
        private void LoadFromServer(
            CharacterAssetData data, Vector3 pos, Quaternion rot, string role,
            Action<GameObject> onCreated)
        {
            var wrapper = new GameObject($"Character_{role}");
            wrapper.transform.SetPositionAndRotation(pos, rot);
            wrapper.transform.localScale = Vector3.one * spawnScale;
            var referenceAvatar = GetReferenceAvatar();

            TripoFbxLoader.Load(
                data.model_url, wrapper,
                onLoaded: () => onCreated?.Invoke(wrapper),
                onError: msg =>
                {
                    Debug.LogWarning($"[CharacterSpawner] 서버 FBX 로드 실패({role}): {msg} → Mock 폴백");
                    Destroy(wrapper);
                    onCreated?.Invoke(InstantiateMock(pos, rot, role));
                },
                referenceAvatar);
        }

        // ════════════════════════════════════════════════════════════
        // 유틸
        // ════════════════════════════════════════════════════════════

        private void SetupAnimatorController(GameObject go)
        {
            if (animatorController == null) return;

            var animator = go.GetComponentInChildren<Animator>(true);
            if (animator == null)
            {
                Debug.LogWarning($"[CharacterSpawner] Animator 없음: {go.name}");
                return;
            }
            if (animator.runtimeAnimatorController == animatorController) return;

            animator.runtimeAnimatorController = animatorController;
        }

        private Avatar GetReferenceAvatar()
        {
            if (mockCharacterPrefab == null) return null;

            var animator = mockCharacterPrefab.GetComponentInChildren<Animator>(true);
            if (animator == null)
            {
                Debug.LogWarning("[CharacterSpawner] mockCharacterPrefab에 Animator가 없어 referenceAvatar를 못 찾음");
                return null;
            }

            var avatar = animator.avatar;
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
            {
                Debug.LogWarning("[CharacterSpawner] mockCharacterPrefab Animator.avatar가 유효한 Humanoid가 아님");
                return null;
            }

            return avatar;
        }

        private static T GetOrAdd<T>(GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            return c != null ? c : go.AddComponent<T>();
        }
    }
}
