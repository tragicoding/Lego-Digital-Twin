# Unity 협업 가이드 (3인: 백엔드 1 · Unity 2)

MINIVERSE / Lego Digital Twin 프로젝트의 협업 규칙. **충돌·파일 누락 방지**가 목적이다.
새 Unity 개발자는 작업 시작 전 이 문서를 끝까지 읽는다.

---

## 0. 대전제 — 환경 통일

- **Unity 버전 100% 동일: `6000.2.2f1`** (`ProjectSettings/ProjectVersion.txt` 기준).
  버전이 다르면 씬·프리팹·에셋이 깨지고 무한 reimport·충돌이 난다. Unity Hub에서 정확히 이 버전 설치.
- 직렬화는 **Force Text**로 고정돼 있다(`EditorSettings: m_SerializationMode: 2`). 절대 Binary로 바꾸지 않는다.
- Version Control Mode = **Visible Meta Files** 유지.

---

## 1. 브랜치 전략

| 담당 | 브랜치 |
|------|--------|
| 백엔드 | `feature/backend` |
| Unity (기존) | `feature/unity` |
| Unity (신규) | `feature/unity-<이름>` → 작업 후 `feature/unity`로 PR 머지 |

- `main` · `develop` **직접 push 금지** — PR + 최소 1인 리뷰로만 머지.
- Unity 두 명이 **같은 브랜치를 동시에 밀면 씬 충돌이 잦다**. 신규 개발자는 개인 브랜치에서 작업하고 자주 `feature/unity`로 합친다.
- 파괴적 명령(`--force`, `reset --hard`, `clean -f`)은 팀 공유 후에만.

---

## 2. 매일 동기화 루틴 (충돌의 80%는 이걸로 예방)

1. **작업 시작 전 반드시 `git pull`** — Unity Editor를 닫은 상태에서.
2. **그날 작업은 그날 push** — 로컬에 며칠 쌓아두면 씬 충돌 폭탄이 된다.
3. push 전 다시 `git pull` → 충돌 없으면 push.
4. 큰 씬/프리팹 작업 시작 전 팀에 공유한다.

---

## 3. 씬·프리팹 충돌 방지 ⭐ (가장 중요)

Unity 2인 협업의 최대 위험은 **`Assets/Scenes/Main.unity` 단일 씬 동시 편집**이다.

- **씬 잠금 규칙**: `Main.unity`는 **한 번에 한 사람만** 편집한다.
  "나 지금 씬 만진다" 공유 → 작업 → 끝나면 즉시 push → 다른 사람이 pull.
- **프리팹 우선 작업**: 기능을 가능한 **프리팹/스크립트로 분리**해 각자 다른 파일을 만진다. 씬에는 배치만 한다.
- **동시 씬 편집이 불가피하면** UnityYAMLMerge(Smart Merge)로 3-way 자동 병합한다(§7 등록 방법).
- 그래도 씬이 충돌나면 **억지로 머지하지 말고** 한쪽이 자기 변경을 다시 적용하는 게 안전하다.

---

## 4. .meta / 파일 누락 방지

- **파일 이동·삭제·이름변경은 반드시 Unity Editor 안에서** 한다. 탐색기에서 하면 `.meta`가 꼬여 GUID 연결이 끊긴다.
- **에셋과 `.meta`는 항상 같이 커밋**한다. 한쪽만 빠지면 다른 사람 프로젝트에서 참조가 깨진다.
- 커밋 전 `git status`로 `.meta` 짝이 맞는지 확인한다.

---

## 5. 절대 커밋 금지 목록

`.gitignore`로 대부분 막혀 있지만 숙지한다:

- `Library/`, `Temp/`, `obj/`, `Build/`, `Builds/`, `Logs/`, `MemoryCaptures/`
- `UserSettings/` (에디터 레이아웃·개인 설정)
- `Assets/TriLib/`, `Assets/TriLibInstaller/` (유료 에셋 — §6)
- `ProjectSettings/ProjectSettings.asset`의 **로컬 `TRILIB` define** (§6)
- `.env` 등 민감 정보

---

## 6. 패키지 · TriLib

- `Packages/manifest.json` 변경(패키지 추가/삭제)은 **한 사람이, 공유 후** 진행한다. 둘이 동시에 하면 `packages-lock.json`이 충돌한다.
- **TriLib 2는 각자 본인 Asset Store 계정으로 설치**한다(저장소엔 올라가지 않음). 미설치여도 `#if TRILIB` 가드로 컴파일되며 캐릭터는 Mock 폴백된다.
- TriLib 설치 시 `Editor/TriLibDefineSetup.cs`가 **머신별로 `TRILIB` define을 자동 추가/제거**한다.
  → 이 때문에 설치 머신에서는 `ProjectSettings.asset`이 항상 `modified`로 보인다. **이 변경은 커밋하지 않는다**(커밋본은 `TRILIB` 비움 = 미설치 환경 컴파일 보장).
- TriLib 설치 후 `Project Settings > TriLib > "Disable in editor glTF2 importing"` 체크(glTFast와 `.glb` 임포터 충돌 방지).

### ProjectSettings · 공유 설정 파일 조율 (충돌 빈발 지점)

`ProjectSettings/` 아래 단일 파일들(`TagManager`·`QualitySettings`·`GraphicsSettings`·`InputManager`·`DynamicsManager`·`URPProjectSettings` 등)은 **모두가 공유하는 하나의 파일**이라, 두 명이 동시에 바꾸면 충돌한다.

- **태그/레이어 추가, 퀄리티·물리·그래픽·입력 설정 변경은 한 명이, 팀 공유 후** 진행한다.
- 변경했으면 **즉시 단독 커밋**(다른 작업과 섞지 않음)하고 알린다 → 다른 사람이 바로 pull.
- `ProjectSettings.asset`의 로컬 `TRILIB` define 변경은 절대 커밋하지 않는다(§6).
- XR/빌드 타깃 전환(Standalone↔Android)도 공유 후 진행한다 — 빌드 설정이 바뀐다.

---

## 7. 신규 개발자 1회 세팅 — UnityYAMLMerge 등록

씬/프리팹 자동 병합(`.gitattributes`의 `merge=unityyamlmerge`)을 쓰려면 각자 로컬 git에 머지 드라이버를 1회 등록한다.

**Windows (Unity Hub 설치 기준):**
```bash
git config merge.unityyamlmerge.name "Unity SmartMerge"
git config merge.unityyamlmerge.driver '"C:/Program Files/Unity/Hub/Editor/6000.2.2f1/Editor/Data/Tools/UnityYAMLMerge.exe" merge -p %O %B %A %A'
git config merge.unityyamlmerge.recursive binary
```

> 설치 경로가 다르면 본인 Unity 버전의 `Editor/Data/Tools/UnityYAMLMerge.exe` 경로로 바꾼다.
> macOS는 `/Applications/Unity/Hub/Editor/6000.2.2f1/Unity.app/Contents/Tools/UnityYAMLMerge`.

---

## 8. 백엔드 ↔ Unity 계약

- API 엔드포인트(`Scripts/Core/ServerConfig.cs`)·`SessionData` 필드 변경은 **양측 합의 후** 진행한다.
- 백엔드가 응답 스키마를 바꾸면 Unity `Resources/Mock/*.json`도 같이 맞춘다.
- Unity 개발자는 평소 **Mock 모드**로 작업한다(`SessionManager.dataSourceMode = Mock`) → 백엔드 서버 없이 독립 개발.
- 통합 테스트 시에만 **Server 모드**로 전환한다.

---

## 9. 커밋 / PR 규칙

- 작은 단위로 자주 커밋. **씬 편집 / 스크립트 / 에셋은 논리 단위로 분리**한다.
- **커밋 전 반드시 `git diff`(또는 `git status`)로 의도한 변경만 들어갔는지 확인**한다 — 특히 씬·프리팹·`ProjectSettings`에 무관한 에디터 자동 변경(오브젝트 흔들림·카메라·선택 상태 등)이 섞이지 않았는지 점검.
- 커밋 메시지 prefix: `feat(unity):` / `fix(unity):` / `chore(unity):` / `docs(unity):`.
- `main` 머지는 PR + 리뷰.
- 작업 기록은 `unity-client/AURA_Lego/Assets/00_Project/History/Claude_MMDD.txt` 또는 `history/` 트러블슈팅 문서에 남긴다.

---

## 10. 충돌이 났을 때 대응 순서

1. `git pull` 시 충돌 발생 → **당황하지 말고** 어떤 파일인지 확인.
2. **스크립트(.cs)** 충돌: 일반 텍스트 머지로 해결.
3. **씬/프리팹/.asset** 충돌: UnityYAMLMerge가 자동 처리. 실패하면 한쪽 변경을 버리고(`git checkout --theirs/--ours`) 그 사람이 Unity에서 다시 적용.
4. **절대 충돌난 씬을 추측으로 손편집하지 않는다** — 깨지면 복구가 어렵다.
5. 해결 후 팀에 공유.

---

## 11. History (작업 기록) 작성 규칙

작업 로그는 두 곳에 남긴다. 작성자가 늘면서 **파일명 충돌**이 나지 않도록 규칙을 정한다.

### Unity 작업 기록 — `Assets/00_Project/History/`
- **기존 개발자(yumin)**: 기존 형식 **`Claude_<MMDD>.txt`** 그대로 사용한다 (예: `Claude_0614.txt`).
- **신규 개발자**: 파일명 **끝에 본인 이름**을 붙인다 → **`Claude_<MMDD>_<이름>.txt`** (예: `Claude_0614_minsu.txt`).
  → 한 사람만 무이름 형식을 쓰고 나머지는 이름을 붙이므로 같은 날 작성해도 파일명이 겹치지 않는다.
- 파일 헤더에 **작성자**를 명시한다:
  ```
  ================================================================
  Claude Code 작업 기록
  날짜: 2026-06-14
  브랜치: feature/unity-minsu
  작성자: minsu
  ================================================================
  ```
- **한 파일 = 한 사람**. 남의 History 파일은 수정하지 않는다(충돌 원천 차단).

### 트러블슈팅 기록 — 루트 `history/`
- 형식: `Trouble_Shooting00.txt`, `Trouble_Shooting01.txt`, … (프로젝트 공통, 모듈 무관).
- 같은 날 두 명이 만들면 번호가 겹칠 수 있으므로, **새 번호를 만들기 전 `git pull`로 최신 번호를 확인**한 뒤 다음 번호로 생성한다.
- 기존 파일은 덮어쓰지 않는다.

### 핵심 3줄
1. 기존 개발자는 `Claude_MMDD.txt`, **신규 개발자는 `Claude_MMDD_이름.txt`** (끝에 이름).
2. **자기 파일만** 쓰고 수정한다.
3. 트러블슈팅 번호는 `git pull` 후 다음 번호로.
