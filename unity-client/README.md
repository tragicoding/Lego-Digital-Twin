# unity-client

레고 놀이공원 VR 환경을 구현하는 Unity 클라이언트입니다.
사용자가 물리 세계에서 조립한 레고 모형이 VR 놀이공원 안에 그대로 재현됩니다.

---

## 프로젝트 구조

```
unity-client/
└── AURA_Lego/          # Unity 프로젝트 루트
    ├── Assets/
    │   ├── Scenes/     # 씬 파일
    │   ├── Scripts/    # C# 스크립트
    │   └── External Assets/  # 외부 에셋 (Skybox 등)
    ├── Packages/
    └── ProjectSettings/
```

---

## 핵심 기능

### VR 놀이공원 환경
- 레고 테마 놀이공원 씬 구성
- 다양한 놀이기구 및 캐릭터 배치
- XR 기반 VR 헤드셋 지원

### 레고 모형 재현
- Backend에서 수신한 `.obj` 파일을 런타임에 로드
- 놀이공원 내 지정 위치에 사용자 모형 배치
- 모형 재질(Material) 자동 적용

### 대기 시스템 (Atmosphere)
Backend로부터 신호를 받아 씬의 대기 상태를 실시간 전환

| 상태 | 트리거 조건 |
|---|---|
| 맑은 낮 | 조도 센서 값 90~100 |
| 노을 | 조도 센서 값 70~89 |
| 흐림 | 조도 센서 값 50~69 |
| 밤 | 조도 센서 값 20~49 |
| 눈 / 비 | 특수 트리거 |

### 인터랙션
- **박수 감지 → 불꽃 축제:** Arduino 박수 신호 수신 시 파티클 이펙트 재생

---

## 기술 스택

| 기술 | 용도 |
|---|---|
| Unity 6000.x | 게임 엔진 |
| C# | 스크립트 |
| Unity XR Toolkit | VR 입력 및 상호작용 |
| Unity Netcode / WebSocket | Backend 통신 |
| Universal Render Pipeline (URP) | 렌더링 |

---

## 개발 환경 설정

1. Unity Hub에서 Unity 6000.x 버전 설치
2. `unity-client/AURA_Lego` 폴더를 Unity Hub에서 프로젝트로 열기
3. XR Plugin Management에서 사용 중인 VR 기기 플랫폼 활성화
4. `Assets/Scripts/NetworkConfig.cs`에서 Backend 서버 주소 설정 (개발 예정)

---

## 개발 브랜치

`feature/unity` 브랜치에서 작업 후 `develop`으로 PR

```bash
# 최초 설정 (팀원 로컬 PC)
git clone https://github.com/tragicoding/Lego-Digital-Twin.git
cd Lego-Digital-Twin
git checkout feature/unity

# 작업 시작 전 develop 최신 내용 반영
git fetch origin
git merge origin/develop

# 작업 후
git add Assets/ Packages/ ProjectSettings/
git commit -m "feat(unity): 작업 내용"
git push origin feature/unity

# GitHub에서 feature/unity → develop PR 생성
```

---

## Git 주의사항

Unity 프로젝트는 자동 생성 파일이 많습니다.

**커밋 포함 대상 (반드시 포함)**
- `Assets/` — 씬, 스크립트, 에셋
- `Assets/**/*.meta` — Unity 에셋 참조에 필수
- `Packages/` — 패키지 목록
- `ProjectSettings/` — 프로젝트 설정

**커밋 제외 대상 (`.gitignore`로 처리됨)**
- `Library/` — Unity가 자동 생성 (용량 수백 MB)
- `Temp/` — 임시 빌드 파일
- `Logs/` — 셰이더 컴파일 로그
- `Obj/` — 빌드 오브젝트
- `Builds/` — 빌드 결과물

> `Library/`는 첫 번째 프로젝트 열기 시 Unity가 자동으로 재생성합니다. 시간이 걸릴 수 있습니다.
