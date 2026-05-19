# unity-client

MINIVERSE 전시용 Unity VR 월드.
관객이 만든 캐릭터 NPC와 오브제가 VR 월드에 등장합니다.

> **Unity는 반드시 Windows에서 실행합니다.**
> WSL/Linux에서 Unity를 실행하지 않습니다.

---

## 프로젝트 열기 (최초 1회)

### Windows에서 클론

```powershell
git clone https://github.com/tragicoding/Lego-Digital-Twin.git
cd Lego-Digital-Twin
git checkout feature/unity
```

### Unity Hub에서 열기

1. Unity Hub 실행
2. **Projects → Add project from disk**
3. 폴더 선택: `Lego-Digital-Twin\unity-client\AURA_Lego`
4. Unity 버전 **6000.2.2f1** 선택
5. Package 다운로드 자동 진행 (5~10분)

---

## 업데이트 (이후)

```powershell
git checkout develop && git pull origin develop
git checkout feature/unity && git pull origin feature/unity
```

---

## 프로젝트 구조

```
unity-client/AURA_Lego/
├── Assets/
│   ├── 00_Project/
│   │   ├── Scripts/
│   │   │   ├── Data/           # SessionData, CharacterAssetData, ObjectAssetData
│   │   │   ├── Network/        # ApiClient (Server Mode)
│   │   │   ├── Character/      # CharacterAnimationController, GuideNPCController
│   │   │   │                   # GeneratedCharacterSpawner
│   │   │   ├── Object/         # GeneratedObjectSpawner
│   │   │   ├── Mock/           # MockSessionLoader
│   │   │   ├── Managers/       # SessionManager, DataSourceMode
│   │   │   ├── Core/           # ServerConfig
│   │   │   └── World/          # BalloonTour, FastSkyDayNight
│   │   └── Resources/
│   │       └── Mock/
│   │           └── mock_session.json   # Mock Mode용 테스트 데이터
│   ├── External Assets/        # 외부 에셋
│   └── StreamingAssets/
│       └── Config/
│           └── server_config.json      # 서버 IP 설정
├── Packages/
└── ProjectSettings/
```

---

## Mock Mode / Server Mode

### Mock Mode (기본값, 팀원 개발용)

- 서버 없이 `Resources/Mock/mock_session.json`을 읽어 동작
- Inspector에서 `SessionManager.dataSourceMode = Mock`
- 서버 연결 불필요

### Server Mode (전시 통합용)

- FastAPI 서버에서 실제 세션 데이터 수신
- Inspector에서 `SessionManager.dataSourceMode = Server`
- `StreamingAssets/Config/server_config.json`에 서버 IP 설정:

```json
{ "base_url": "http://192.168.x.x:8000" }
```

---

## 씬에 추가해야 할 컴포넌트

`SessionManager`, `ApiClient` 컴포넌트를 씬의 GameObject에 추가해야 합니다.

| 컴포넌트 | 역할 |
|---|---|
| `SessionManager` | Mock/Server 모드 전환, 세션 데이터 로드 |
| `ApiClient` | Server Mode에서 FastAPI 통신 |
| `GuideNPCController` | 캐릭터 이동, 말풍선, 시나리오 |
| `CharacterAnimationController` | animation_walk(), animation_idle() |
| `GeneratedCharacterSpawner` | 캐릭터 생성 |
| `GeneratedObjectSpawner` | 오브제 생성 |

---

## 캐릭터 애니메이션 사용법

```csharp
// 팀원 시나리오 작성 예시
npc.animation_walk();
npc.animation_idle();
npc.PlayAnimation("walk");

// NPC 이동
guideNPC.MoveTo(targetPosition);
guideNPC.SetBubbleText("MINIVERSE에 온 걸 환영해!");
guideNPC.StartGuideScenario();
```

---

## Git 전략 (Unity 개발자)

```bash
git checkout feature/unity
git fetch origin && git merge origin/develop
# Unity Editor에서 작업
# 파일 이동은 반드시 Unity Editor 내에서 진행 (.meta 자동 관리)
git add <파일명> <파일명.meta>
git commit -m "feat(unity): ..."
git push origin feature/unity
# GitHub에서 feature/unity → develop PR 생성
```

### 커밋 금지 항목

- `Library/`, `Temp/`, `obj/`, `Logs/` 폴더
- `.env` 파일
- 대용량 바이너리 (별도 정책 필요 시 Git LFS 사용)

### 주의사항

- Unity 파일(`*.unity`, `*.prefab`, `*.asset`) 이동 시 **반드시 Unity Editor 내에서** 이동한다.
- `.meta` 파일을 직접 삭제하거나 이동하지 않는다.
- `ProjectSettings/`, `Packages/` 변경은 담당자와 협의 후 진행한다.
