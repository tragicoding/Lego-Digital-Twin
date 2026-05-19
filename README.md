# Lego Digital Twin / MINIVERSE

> 관객이 현실에서 조립한 레고 캐릭터와 오브제를 촬영하면,
> 3D 모델로 변환되어 Unity VR 월드에 등장하는 디지털 트윈 전시 프로젝트

---

## 프로젝트 개요

관객이 태블릿 UI를 통해 자신의 레고 **캐릭터**와 **오브제**를 촬영합니다.
촬영된 이미지는 Tripo3D API로 3D 모델로 변환됩니다.

- **character**: 3D 변환 + walk/idle 애니메이션 적용 → Unity VR 월드에서 NPC로 등장
- **object**: 3D 변환만 수행 → Unity VR 월드에 정적 3D 오브제로 배치

```
[관객]
  촬영 (태블릿 UI)
      │
      ▼
[Frontend]  →  POST /sessions/{id}/assets
      │
      ▼
[Backend / FastAPI]
  - Redis Queue 등록
  - Tripo3D API: image_to_model → rig → retarget (walk + idle 병렬)
  - GLB 저장
      │
      ▼
[Unity VR 월드 (Windows)]
  - WebSocket session_ready 이벤트 수신
  - GLB 다운로드 및 로드
  - 캐릭터 NPC + 오브제 배치
```

---

## 팀 구성

| 이름 | 소속 | 역할 |
|---|---|---|
| 김예진 | 중앙대학교 | Project Manager / Team Leader |
| 김진서 | 중앙대학교 | Software Engineer / Lead Developer |
| 김유민 | 중앙대학교 | Unity Developer / Lead Designer |
| 이서연 | 중앙대학교 | Animator / Sub Project Manager |

**Develop Role:**
- System Architect · Backend · Frontend · System Automation: **김진서**
- Unity Design · Unity Script: **김유민**

---

## 개발 환경

| 파트 | OS | 주요 도구 |
|---|---|---|
| Backend / Frontend | Linux / WSL Ubuntu | Python 3.11, FastAPI, Node.js, Docker |
| Unity | **Windows** | Unity Hub, Unity Editor 6000.2.2f1 |
| Git 협업 | 공통 | GitHub, feature branch + PR |

> **중요**: Unity는 반드시 **Windows**에서 실행한다.
> WSL/Linux에서 Unity 실행을 시도하지 않는다.

---

## 디렉토리 구조

```
Lego-Digital-Twin/
├── apps/
│   ├── backend/        # FastAPI 서버, PostgreSQL, Redis/RQ Worker, Tripo API
│   ├── frontend/       # 태블릿 UI (React + Vite + TypeScript)
│   └── hardware/       # Arduino 제어 (예비)
├── unity-client/       # Unity VR 월드 (C#, XR, Mock/Server Mode)
├── docs/               # 환경 설정 문서
├── scripts/            # 자동화 스크립트
├── history/            # 작업 기록 · Trouble Shooting
├── docker-compose.yml  # PostgreSQL 16 + Redis 7
├── run_worker.py       # RQ Worker 시작 스크립트 (character + object 병렬)
├── README.md
├── git.md              # Git 전략 (Single Source of Truth)
└── CLAUDE.md           # Claude Code 설정
```

---

## 실행 방법 (Backend + Frontend)

> Linux / WSL Ubuntu 기준. conda 환경: `triposr`

### 1. 의존성 설치

```bash
# Backend
conda activate triposr
pip install -r apps/backend/requirements.txt

# Frontend
cd apps/frontend && npm install
```

### 2. 환경 변수 설정

```bash
cp apps/backend/.env.example apps/backend/.env
# .env 파일에 TRIPO_API_KEY, DATABASE_URL, REDIS_URL 입력
```

### 3. Docker (PostgreSQL + Redis) 실행

```bash
docker compose up -d db redis
```

### 4. FastAPI 서버 실행

```bash
conda run -n triposr uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000
```

### 5. RQ Worker 실행

```bash
conda run -n triposr python run_worker.py
```

### 6. Frontend 실행

```bash
cd apps/frontend && npm run dev -- --host 0.0.0.0 --port 3000
```

### 7. Unity 실행 (Windows)

Windows에서 별도 진행. [unity-client/README.md](unity-client/README.md) 참고.

---

## Backend Developer가 Unity 테스트할 때

최초 1회:
```bash
# Windows PowerShell / Git Bash
git clone https://github.com/tragicoding/Lego-Digital-Twin.git
cd Lego-Digital-Twin
git checkout feature/unity
```

이후:
```bash
git checkout develop && git pull origin develop
git checkout feature/unity && git pull origin feature/unity
```

Unity Hub → Add project from disk → `unity-client/AURA_Lego/` 선택

---

## 브랜치 전략

```
main                    ← 최종 안정 버전 (직접 push 금지)
└── develop             ← 통합 브랜치 (모든 PR 대상)
    ├── feature/unity       ← Unity 개발 (Windows)
    ├── feature/backend     ← Backend 개발 (Linux/WSL)
    ├── feature/frontend    ← Frontend 개발 (Linux/WSL)
    ├── feature/system      ← System 자동화
    └── test_app            ← 실험·기획 검증
```

자세한 규칙은 [git.md](git.md) 참고.

---

## 협업 원칙

- `develop`은 통합 브랜치다. 직접 push하지 않는다.
- 모든 기능 작업은 feature 브랜치에서 진행한다.
- 작업 전 `develop` 최신 내용을 pull한다.
- PR 생성 후 리뷰를 거쳐 merge한다.
- `.env` 파일은 커밋하지 않는다.
- Unity의 `Library/`, `Temp/`, `obj/`, `Logs/`는 커밋하지 않는다.
- Unity 파일 이동 시 `.meta` 파일을 반드시 함께 관리한다.
