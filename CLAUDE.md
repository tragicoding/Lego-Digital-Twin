# CLAUDE.md — Lego Digital Twin / MINIVERSE

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 자동으로 로드하는 설정입니다.

---

## 1. Claude 작업 기본 원칙

- 작업 시작 전 `README.md`, `git.md`, `CLAUDE.md`를 먼저 읽는다.
- 현재 디렉토리 구조를 확인한 뒤 작업한다.
- 사용자 요청 범위를 벗어난 대규모 리팩토링을 하지 않는다.
- 기존 테스트 완료 흐름을 깨지 않는다.
- 불확실한 부분은 TODO 또는 질문으로 남긴다.
- 실제 파일 구조를 확인하지 않고 임의로 문서화하지 않는다.

---

## 2. Git 작업 원칙

**모든 git 작업은 [git.md](git.md)의 규칙을 따른다.**

- 작업 전 `git status` 확인
- **commit 전 사용자에게 반드시 확인**
- **push 전 사용자에게 반드시 확인**
- PR 생성 전 사용자에게 반드시 확인
- 사용자의 명시적 승인 없이 commit/push하지 않는다.
- 사용자의 명시적 승인 없이 merge/rebase하지 않는다.
- `git add .` 허용 — commit 전 `git status`로 포함 파일 반드시 확인한다.
- push 대상: `feature/*` 또는 `test_app` 브랜치만 허용
- `main`·`develop` 직접 push 금지
- 파괴적 명령 (`--force`, `reset --hard`, `clean -f` 등): 사용자 확인 필수
- **브랜치 이탈 감지**: 현재 브랜치가 `feature/*` 또는 `test_app`이 아니면 작업 전 사용자에게 알림

---

## 3. 프로젝트 구조

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
├── docker-compose.yml
├── run_worker.py       # RQ Worker 시작 (lego-character + lego-object 병렬)
├── README.md
├── git.md
└── CLAUDE.md
```

---

## 4. 기술 스택

- **Backend**: Python 3.11, FastAPI, PostgreSQL 16, Redis 7, RQ, Tripo3D API
- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS, Framer Motion, Zustand
- **Unity**: Unity 6000.2.2f1, C#, Meta XR SDK, glTFast
- **환경**: Linux / WSL2 (Ubuntu), Docker, conda 환경: `triposr`

---

## 5. Unity 작업 원칙

- Unity는 **Windows**에서 실행한다. WSL/Linux에서 실행하지 않는다.
- Unity 개발자는 `feature/unity` 브랜치를 사용한다.
- Unity 파일 이동은 반드시 **Unity Editor 내에서** 진행한다 (`.meta` 자동 관리).
- `Library/`, `Temp/`, `obj/`, `Logs/` 커밋 금지.
- Mock Mode와 Server Mode를 분리한다:
  - **Mock Mode**: 서버 없이 `Resources/Mock/mock_session.json` 사용 (기본값, 팀원 개발용)
  - **Server Mode**: FastAPI 서버 데이터 수신 (전시 통합 테스트)

---

## 6. Backend / Frontend 작업 원칙

- Linux / WSL Ubuntu 기준으로 개발한다.
- conda 환경: `triposr`
- Backend: FastAPI, Redis/RQ Worker, PostgreSQL, Tripo API 연동 중심
- Frontend: 태블릿 UI, 촬영, 이름 입력, 로딩 UX 중심
- Docker / requirements / package.json 변경 시 README도 함께 수정한다.

---

## 7. 문서 자동 업데이트 원칙

다음 상황이 발생하면 관련 README를 수정해야 한다:

- 디렉토리 구조 변경
- 브랜치 전략 변경
- 실행 방법 변경
- Docker / requirements / package.json 의존성 변경
- Unity 프로젝트 실행 방식 변경
- backend / frontend / unity-client 역할 변경
- 전시 기획 변경 (촬영 대상, 처리 방식 등)
- Mock Mode / Server Mode 변경
- Trouble Shooting 기록이 생긴 경우

---

## 8. Trouble Shooting 기록 원칙

문제 해결 경험이 생길 때마다 `history/` 폴더에 Trouble Shooting 문서를 작성한다.

**규칙:**
- 저장 위치: `history/`
- 파일명 형식: `Trouble_Shooting00.txt`, `Trouble_Shooting01.txt`, ...
- 기존 파일 번호를 확인하고 다음 번호로 생성한다.
- 이미 존재하는 파일을 덮어쓰지 않는다.

**포함 내용:**
```
# Trouble Shooting XX

## Date
YYYY-MM-DD

## Environment
- OS:
- Branch:
- Module:

## Problem
문제 상황 설명

## Error Message
에러 메시지

## Cause
원인 분석

## Solution
해결 방법

## Prevention
재발 방지 방법

## Related Files / Commands
관련 파일 및 명령어

## Notes
추가 참고 사항
```

---

## 9. 개발자 확인 필요 조건

Claude는 다음 행동 전에 반드시 사용자에게 확인해야 한다:

- commit, push
- 브랜치 생성·삭제
- merge, rebase
- 패키지 설치 (npm install, pip install)
- 대규모 파일 이동·삭제
- 기존 구조 삭제
- `ProjectSettings/`, `Packages/` 변경
- Unity Scene 변경
- `.env` 또는 민감 정보 관련 변경

---

## 10. 작업 완료 후 보고 형식

작업 완료 후 다음을 요약한다:

- 수정한 파일
- 변경 내용
- 테스트 방법
- 추가로 필요한 작업
- Trouble Shooting 문서 생성 여부
- commit 여부 확인 질문
