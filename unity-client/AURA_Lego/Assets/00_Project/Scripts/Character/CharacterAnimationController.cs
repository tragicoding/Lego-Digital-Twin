using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using LegoTwin.Data;

namespace LegoTwin.Character
{
    /// <summary>
    /// 캐릭터 애니메이션 컨트롤러.
    /// 캐릭터 루트 GameObject에 추가한다.
    ///
    /// 팀원 사용 예:
    ///   npc.Animation.animation_walk();
    ///   npc.Animation.animation_idle();
    ///   npc.Animation.PlayAnimation("walk");
    ///   npc.Animation.PlayMotionClip(clip);   // Mixamo 클립 런타임 교체
    /// </summary>
    public class CharacterAnimationController : MonoBehaviour
    {
        private static readonly Dictionary<string, Material[]> MaterialCache = new();
        private static readonly Dictionary<string, Task<Material[]>> MaterialLoadTasks = new();

        [Header("NPC Info")]
        public string npcName;
        public string bubbleText;

        [Header("Animator 파라미터")]
        [Tooltip("Bool 방식 파라미터 이름 (Animator에 이 Bool이 있으면 우선 사용)")]
        public string walkBoolParam = "isWalking";
        [Tooltip("Trigger 방식 — 걷기 Trigger 이름 (Bool 없을 때 사용)")]
        public string walkTrigger   = "walk";
        [Tooltip("Trigger 방식 — 정지 Trigger 이름 (Bool 없을 때 사용)")]
        public string idleTrigger   = "idle";

        private Animator _animator;
        private AnimatorOverrideController _overrideController;
        private bool _useBoolParam;
        private Coroutine _loopCoroutine;
        private System.Action _onLoopRestart;

        // AnimatorOverrideController에서 교체할 클립 슬롯 이름
        // Base Animator Controller의 "Motion" 스테이트 클립 이름과 일치해야 함
        private const string MOTION_SLOT = "Motion";

        private void Awake()
        {
            _animator = GetComponentInChildren<Animator>();
            if (_animator == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] Animator 없음: {gameObject.name}");
                return;
            }
            if (_animator.runtimeAnimatorController == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] Animator Controller 미연결: {gameObject.name}\n" +
                                 "Inspector에서 Animator Controller를 연결해야 애니메이션이 재생됩니다.\n" +
                                 "물리적 이동(transform)은 Controller 없이도 동작합니다.");
                return;
            }
            InitOverrideController();
            DetectAnimatorMode();
        }

        /// <summary>
        /// AnimatorOverrideController 초기화.
        /// Base Animator Controller가 설정되어 있어야 동작한다.
        /// </summary>
        private void InitOverrideController()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null)
                return;

            // 이미 Override라면 중복 생성 방지
            if (_animator.runtimeAnimatorController is AnimatorOverrideController)
            {
                _overrideController = _animator.runtimeAnimatorController as AnimatorOverrideController;
                return;
            }

            _overrideController = new AnimatorOverrideController(_animator.runtimeAnimatorController);
            _animator.runtimeAnimatorController = _overrideController;
        }

        /// <summary>
        /// Animator 파라미터를 확인하여 Bool / Trigger 방식을 자동 감지합니다.
        /// </summary>
        private void DetectAnimatorMode()
        {
            foreach (var p in _animator.parameters)
            {
                if (p.name == walkBoolParam && p.type == AnimatorControllerParameterType.Bool)
                {
                    _useBoolParam = true;
                    return;
                }
            }
        }

        // ── 공개 애니메이션 함수 ─────────────────────────────────────

        /// <summary>걷기 애니메이션 (Bool / Trigger 자동 분기)</summary>
        public void animation_walk()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            if (_useBoolParam) _animator.SetBool(walkBoolParam, true);
            else               PlayAnimation(walkTrigger);
        }

        /// <summary>idle(대기) 애니메이션 (Bool / Trigger 자동 분기)</summary>
        public void animation_idle()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            if (_useBoolParam) _animator.SetBool(walkBoolParam, false);
            else               PlayAnimation(idleTrigger);
        }

        /// <summary>animation key 기반 Trigger 실행</summary>
        public void PlayAnimation(string animationKey)
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            _animator.SetTrigger(animationKey);
        }

        /// <summary>
        /// 가이드 등장 걷기 등 Animator 'Walk' 상태의 클립을 모션 라이브러리의 Walk 클립으로 교체한다.
        ///
        /// 가이드 걷기는 원래 Animator Controller 의 Walk 상태가 .anim 파일을 직접 참조한다.
        /// 그 파일을 바꾸거나 지우면 등장 걷기가 누락되므로, 모션 라이브러리(MotionType.Walk)를
        /// 단일 소스로 삼아 런타임에 Walk 슬롯을 덮어쓴다.
        ///   → 라이브러리의 Walk 클립만 바꾸면 가이드 걷기도 함께 바뀌고,
        ///     특정 Walk 클립 파일이 사라져도 라이브러리에 Walk 가 있으면 누락되지 않는다.
        ///
        /// 동작: AnimatorOverrideController 에서 이름이 "Walk" 로 시작하는 슬롯(예: Walk2)을
        ///       찾아 라이브러리 Walk 클립으로 덮어쓴다. 라이브러리에 Walk 가 없으면 아무것도
        ///       하지 않아 Animator 의 기본 Walk 클립이 폴백으로 그대로 사용된다.
        /// </summary>
        public void ApplyWalkFromLibrary(MixamoMotionLibrary library)
        {
            if (library == null) return;
            if (_overrideController == null) InitOverrideController();
            if (_overrideController == null) return;

            // 라이브러리에 Walk 가 없으면 base 클립(폴백)을 그대로 둔다.
            if (!library.HasClip(MotionType.Walk)) return;
            var walkClip = library.GetClip(MotionType.Walk);
            if (walkClip == null) return;

            var overrides = new List<KeyValuePair<AnimationClip, AnimationClip>>();
            _overrideController.GetOverrides(overrides);

            bool changed = false;
            for (int i = 0; i < overrides.Count; i++)
            {
                var baseClip = overrides[i].Key;
                if (baseClip != null &&
                    baseClip.name.StartsWith("Walk", System.StringComparison.OrdinalIgnoreCase))
                {
                    overrides[i] = new KeyValuePair<AnimationClip, AnimationClip>(baseClip, walkClip);
                    changed = true;
                }
            }

            if (changed) _overrideController.ApplyOverrides(overrides);
        }

        /// <summary>
        /// Mixamo AnimationClip을 런타임에 교체하여 재생한다.
        /// AnimatorOverrideController의 "Motion" 슬롯을 교체한 뒤
        /// "motion" Trigger를 발생시킨다.
        ///
        /// 유니티 개발자 체크리스트:
        ///   [ ] Base Animator Controller에 "Motion" 스테이트 추가
        ///       (Any State → Motion, Trigger 파라미터: "motion")
        /// </summary>
        public void PlayMotionClip(AnimationClip clip)
        {
            if (clip == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] clip이 null입니다.");
                return;
            }

            if (_overrideController == null)
            {
                Debug.LogWarning($"[CharacterAnimationController] OverrideController 없음 — " +
                                 $"Base Animator Controller가 설정되어 있는지 확인하세요.");
                return;
            }

            _overrideController[MOTION_SLOT] = clip;
            _animator.SetTrigger("motion");
        }

        /// <summary>
        /// Mixamo 클립을 무한 루프 재생한다.
        /// normalizedTime이 0.95에 도달하면 같은 상태를 처음부터 재시작해
        /// Idle로 빠져나가지 않고 매끄럽게 반복한다.
        /// StopMotionLoop() 또는 animation_idle()로 중단한다.
        /// </summary>
        /// <summary>
        /// Mixamo 클립을 무한 루프 재생한다.
        /// onLoopRestart: 루프가 재시작될 때마다 호출되는 콜백 (위치 복원 등에 사용).
        /// </summary>
        public void PlayMotionClipLooping(AnimationClip clip, System.Action onLoopRestart = null)
        {
            if (clip == null || _overrideController == null) return;

            if (_loopCoroutine != null)
                StopCoroutine(_loopCoroutine);

            _onLoopRestart = onLoopRestart;
            _overrideController[MOTION_SLOT] = clip;
            // 초기 진입은 SetTrigger — Play()는 첫 프레임 포즈로 스냅해 90도 회전 발생
            // 루프 재시작(LoopRoutine 내)은 이미 Motion 상태이므로 Play()를 사용해도 무관
            _animator.SetTrigger("motion");
            _loopCoroutine = StartCoroutine(LoopRoutine());
        }

        /// <summary>루프 모션을 중단하고 idle 상태로 복귀한다.</summary>
        public void StopMotionLoop()
        {
            if (_loopCoroutine != null)
            {
                StopCoroutine(_loopCoroutine);
                _loopCoroutine = null;
            }
            animation_idle();
        }

        private System.Collections.IEnumerator LoopRoutine()
        {
            // Motion 상태 진입 대기
            yield return new WaitUntil(() =>
                _animator.GetCurrentAnimatorStateInfo(0).IsName("Motion"));

            while (true)
            {
                // normalizedTime이 0.95에 도달하면 처음부터 재시작 → 끊김 없이 반복
                yield return new WaitUntil(() =>
                {
                    var info = _animator.GetCurrentAnimatorStateInfo(0);
                    return info.IsName("Motion") && info.normalizedTime >= 0.95f;
                });

                _onLoopRestart?.Invoke();   // 루프 재시작 전 위치 복원 등 외부 콜백 실행
                _animator.Play("Motion", 0, 0f);
                yield return null;
            }
        }

        // ── 초기화 ───────────────────────────────────────────────────

        /// <summary>CharacterAssetData로 NPC 정보를 초기화한다.</summary>
        public void Initialize(CharacterAssetData data, string bubble = "")
        {
            if (data == null) return;
            npcName    = data.npc_name;
            bubbleText = bubble;

            // Server Mode: texture_url(PBR GLB)이 있으면 SkinnedMeshRenderer에 머티리얼 적용
            if (!string.IsNullOrEmpty(data.texture_url))
                ApplyTextureFromUrl(data.texture_url);
        }

        private async void ApplyTextureFromUrl(string url)
        {
            if (IsImageUrl(url))
            {
                var imageMaterial = await LoadImageMaterial(url);
                if (imageMaterial == null)
                {
                    Debug.LogWarning($"[CharacterAnimationController] {npcName}: 이미지 텍스쳐 로드 실패: {url}");
                    return;
                }

                foreach (var dstRenderer in GetComponentsInChildren<SkinnedMeshRenderer>(true))
                    dstRenderer.sharedMaterial = imageMaterial;
                return;
            }

            Material[] sharedMaterials = await GetOrLoadSharedMaterials(url);
            if (sharedMaterials == null || sharedMaterials.Length == 0)
            {
                Debug.LogWarning($"[CharacterAnimationController] {npcName}: 텍스쳐 GLB 로드 실패: {url}");
                return;
            }

            var dstRenderers = GetComponentsInChildren<SkinnedMeshRenderer>(true);
            if (dstRenderers.Length == 0)
            {
                Debug.LogWarning($"[CharacterAnimationController] {npcName}: SkinnedMeshRenderer 없음 — 텍스쳐 적용 생략");
                return;
            }

            foreach (var dstRenderer in dstRenderers)
                dstRenderer.sharedMaterials = sharedMaterials;
        }

        private static bool IsImageUrl(string url)
        {
            string lower = url.ToLowerInvariant();
            return lower.EndsWith(".jpg") || lower.EndsWith(".jpeg") || lower.EndsWith(".png");
        }

        private static async Task<Material> LoadImageMaterial(string url)
        {
            using var req = UnityWebRequestTexture.GetTexture(url);
            var op = req.SendWebRequest();
            while (!op.isDone)
                await Task.Yield();

            if (req.result != UnityWebRequest.Result.Success)
                return null;

            var texture = DownloadHandlerTexture.GetContent(req);
            var material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            material.mainTexture = texture;
            return material;
        }

        private static Task<Material[]> GetOrLoadSharedMaterials(string url)
        {
            if (string.IsNullOrEmpty(url))
                return Task.FromResult<Material[]>(null);

            if (MaterialCache.TryGetValue(url, out var cached))
                return Task.FromResult(cached);

            if (MaterialLoadTasks.TryGetValue(url, out var inFlight))
                return inFlight;

            var task = LoadSharedMaterials(url);
            MaterialLoadTasks[url] = task;
            return task;
        }

        private static async Task<Material[]> LoadSharedMaterials(string url)
        {
            try
            {
                var gltf = new GLTFast.GltfImport();
                bool ok = await gltf.Load(url);
                if (!ok) return null;

                // GLB 씬을 비활성 임시 컨테이너에 인스턴스화해 머티리얼 추출
                var container = new GameObject("_TempGltfMaterial");
                container.SetActive(false);
                await gltf.InstantiateMainSceneAsync(container.transform);

                var srcRenderer = container.GetComponentInChildren<Renderer>();
                if (srcRenderer == null || srcRenderer.sharedMaterials.Length == 0)
                {
                    UnityEngine.Object.Destroy(container);
                    return null;
                }

                var clonedMaterials = new Material[srcRenderer.sharedMaterials.Length];
                for (int i = 0; i < srcRenderer.sharedMaterials.Length; i++)
                {
                    var source = srcRenderer.sharedMaterials[i];
                    clonedMaterials[i] = source != null ? new Material(source) : null;
                }

                MaterialCache[url] = clonedMaterials;
                UnityEngine.Object.Destroy(container);
                return clonedMaterials;
            }
            finally
            {
                MaterialLoadTasks.Remove(url);
            }
        }

        /// <summary>
        /// 텍스처 머티리얼 캐시를 비운다. 새 세션 시작 시(GameFlowManager.Awake) 호출.
        ///
        /// MaterialCache는 static이라 씬 리로드로 자동 정리되지 않는다. 관람객마다 새 texture_url이
        /// 쌓이면 클론 머티리얼·텍스처 참조가 끊기지 않아 UnloadUnusedAssets로도 회수되지 못한다.
        /// 세션 경계에서 캐시의 머티리얼·메인 텍스처를 명시적으로 파괴해 누수를 막는다.
        /// (이 시점엔 이전 세션 캐릭터가 이미 파괴돼 캐시 외 참조가 없으므로 안전.)
        /// </summary>
        /// <summary>(디버그/검증용) 현재 캐시된 텍스처 머티리얼 세트 수. 세션마다 누적되지 않아야 정상.</summary>
        public static int CachedMaterialSetCount => MaterialCache.Count;

        public static void ClearMaterialCache()
        {
            foreach (var mats in MaterialCache.Values)
            {
                if (mats == null) continue;
                foreach (var m in mats)
                {
                    if (m == null) continue;
                    if (m.mainTexture != null) UnityEngine.Object.Destroy(m.mainTexture);
                    UnityEngine.Object.Destroy(m);
                }
            }
            MaterialCache.Clear();
            MaterialLoadTasks.Clear();
        }
    }
}
