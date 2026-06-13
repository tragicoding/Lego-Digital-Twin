using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace LegoTwin.Core
{
    /// <summary>
    /// 성능 효과 확인용 디버그 HUD (전시 빌드에서는 비활성/제거).
    ///
    /// ── 사용법 ──────────────────────────────────────────────────────
    ///  빈 GameObject에 부착 후 Play. 좌상단에 FPS·프레임 ms가 뜬다.
    ///  에디터에서는 삼각형/버텍스/SetPass/드로우콜/그림자 캐스터 수도 표시.
    ///
    ///  F9 : 씬 전체 렌더러에 RuntimeOptimizer 최적화를 즉시 켜고/끈다(A/B).
    ///       (최적화 OFF→ON 순간 FPS·드로우콜이 어떻게 변하는지 바로 비교)
    ///  F10: 일시정지(Time.timeScale 0↔1) — 한 프레임 수치 정밀 비교용
    ///
    /// ── 주의 ────────────────────────────────────────────────────────
    ///  Revert는 "원본이 ShadowCasting.On / SkinQuality.Auto"라고 가정한다
    ///  (임포트 기본값). A/B 비교 용도로만 쓰고, 실제 최적화는 스폰 시
    ///  RuntimeOptimizer가 자동 적용한다.
    /// </summary>
    public class PerfHud : MonoBehaviour
    {
        [Tooltip("F9 토글 시 씬 전체 렌더러에 적용/해제")]
        public KeyCode toggleKey = KeyCode.F9;
        [Tooltip("F10 시간 정지 토글")]
        public KeyCode pauseKey = KeyCode.F10;

        private float _deltaSmooth;
        private bool _optimized = true;   // 스폰 시 이미 최적화된 상태로 시작
        private GUIStyle _style;

        private void Update()
        {
            _deltaSmooth += (Time.unscaledDeltaTime - _deltaSmooth) * 0.1f;

            if (Input.GetKeyDown(toggleKey))
            {
                _optimized = !_optimized;
                if (_optimized) ApplyAll();
                else            RevertAll();
            }

            if (Input.GetKeyDown(pauseKey))
                Time.timeScale = Time.timeScale > 0f ? 0f : 1f;
        }

        private void OnGUI()
        {
            if (_style == null)
            {
                _style = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 18,
                    normal = { textColor = Color.white }
                };
            }

            float ms = _deltaSmooth * 1000f;
            float fps = _deltaSmooth > 0f ? 1f / _deltaSmooth : 0f;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"FPS {fps:F0}  ({ms:F1} ms)");
            sb.AppendLine($"최적화(F9): {(_optimized ? "ON" : "OFF")}   정지(F10): {(Time.timeScale == 0f ? "ON" : "OFF")}");

#if UNITY_EDITOR
            sb.AppendLine($"Tris {UnityEditor.UnityStats.triangles:N0}  Verts {UnityEditor.UnityStats.vertices:N0}");
            sb.AppendLine($"SetPass {UnityEditor.UnityStats.setPassCalls}  DrawCalls {UnityEditor.UnityStats.drawCalls}");
            sb.AppendLine($"Batches {UnityEditor.UnityStats.batches}  ShadowCasters {UnityEditor.UnityStats.shadowCasters}");
#endif

            // 배경 박스 + 텍스트
            GUI.color = new Color(0f, 0f, 0f, 0.5f);
            GUI.DrawTexture(new Rect(8, 8, 360, 150), Texture2D.whiteTexture);
            GUI.color = Color.white;
            GUI.Label(new Rect(16, 12, 360, 150), sb.ToString(), _style);
        }

        // ── 씬 전체 A/B 토글 ─────────────────────────────────────────

        private static IEnumerable<Renderer> SceneRenderers()
        {
            return FindObjectsByType<Renderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        }

        private void ApplyAll()
        {
            foreach (var r in SceneRenderers())
            {
                if (r == null) continue;
                if (RuntimeOptimizer.disableShadowCasting)
                    r.shadowCastingMode = ShadowCastingMode.Off;
                if (r is SkinnedMeshRenderer smr && RuntimeOptimizer.limitSkinQuality)
                {
                    smr.quality = RuntimeOptimizer.skinQuality;
                    if (RuntimeOptimizer.disableSkinnedMotionVectors)
                        smr.skinnedMotionVectors = false;
                }
            }
        }

        private void RevertAll()
        {
            foreach (var r in SceneRenderers())
            {
                if (r == null) continue;
                r.shadowCastingMode = ShadowCastingMode.On;
                if (r is SkinnedMeshRenderer smr)
                {
                    smr.quality = SkinQuality.Auto;
                    smr.skinnedMotionVectors = true;
                }
            }
        }
    }
}
