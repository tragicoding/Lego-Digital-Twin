using System.IO;
using UnityEngine;
using LegoTwin.Data;

namespace LegoTwin.Plaza
{
    /// <summary>
    /// Mock 광장 데이터(mock_plaza.json) 영속 저장소.
    ///
    /// ── 핵심 ───────────────────────────────────────────────────────
    ///  Editor·Build 모두 persistentDataPath 의 '가변 복사본' 을 읽고 쓴다.
    ///  Resources/Mock/mock_plaza.json 은 '읽기 전용 시드' 로만 사용(최초 1회 복사).
    ///  → 런타임 변경이 추적되는 Assets 파일을 더럽히지 않아, 플레이테스트마다
    ///    git 에 mock_plaza.json 이 modified 로 뜨던 문제가 근본 해소된다.
    ///
    /// ── 책임 격리 ──────────────────────────────────────────────────
    ///  경로·시딩·직렬화를 이 클래스가 전담한다. PlazaManager 는 Load()/Save() 만
    ///  호출하며 파일 시스템·JSON·플랫폼 분기를 알 필요가 없다(의존성 최소화).
    ///  저장 위치를 바꾸거나 백엔드를 교체할 때도 이 파일만 수정하면 된다.
    ///
    /// ── 시드 리셋 ──────────────────────────────────────────────────
    ///  persistentDataPath 의 mock_plaza.json 을 삭제하면 다음 실행에서 시드로 재생성.
    ///  (Editor 콘솔에서 경로 확인: Debug.Log(MockPlazaStore.FilePath))
    /// </summary>
    internal static class MockPlazaStore
    {
        // Resources 기준 경로(확장자 제외). 시드 원본.
        private const string SeedResourcePath = "Mock/mock_plaza";

        /// <summary>가변 복사본 경로. Editor·Build 공통(추적 Assets 파일은 절대 쓰지 않음).</summary>
        public static string FilePath =>
            Path.Combine(Application.persistentDataPath, "mock_plaza.json");

        /// <summary>가변 복사본이 없으면 Resources 시드에서 1회 생성한다.</summary>
        public static void EnsureSeeded()
        {
            if (File.Exists(FilePath)) return;

            var seed = Resources.Load<TextAsset>(SeedResourcePath);
            if (seed == null)
            {
                Debug.LogError($"[MockPlazaStore] 시드 없음: Resources/{SeedResourcePath}.json");
                return;
            }
            File.WriteAllText(FilePath, seed.text);
        }

        /// <summary>광장 데이터를 로드한다. 시드 보장 후 읽으며, 실패 시 null.</summary>
        public static PlazaResponse Load()
        {
            EnsureSeeded();
            if (!File.Exists(FilePath))
            {
                Debug.LogError("[MockPlazaStore] mock_plaza.json 로드 실패");
                return null;
            }
            return JsonUtility.FromJson<PlazaResponse>(File.ReadAllText(FilePath));
        }

        /// <summary>광장 데이터를 저장한다(가변 복사본에만 기록).</summary>
        public static void Save(PlazaResponse plaza)
        {
            if (plaza == null) return;
            EnsureSeeded();
            File.WriteAllText(FilePath, JsonUtility.ToJson(plaza, prettyPrint: true));
        }
    }
}
