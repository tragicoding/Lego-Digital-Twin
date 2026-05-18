# CLAUDE.md — Lego Digital Twin

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 자동으로 로드하는 설정입니다.

---

## Git 전략 자동 적용

**모든 git 작업은 [git.md](git.md)의 규칙을 따른다.**

Claude는 git 관련 작업 수행 시 git.md의 AGENT-RULES 섹션을 자동으로 적용한다:

- **커밋 전**: `git fetch origin develop && git merge origin/develop` 실행 여부 확인
- **커밋 메시지**: 반드시 `<type>(<scope>): <subject>` 형식 사용
- **스테이징**: `git add <파일명>` 으로 명시적 지정, `add .` 사용 금지
- **push 대상**: `feature/*` 브랜치만 허용, `main`·`develop` 직접 push 금지
- **파괴적 명령** (`--force`, `reset --hard`, `clean -f` 등): 사용자 확인 필수
- **PR**: `develop` 브랜치 대상으로 생성
- **브랜치 이탈 감지**: 현재 브랜치가 `feature/*`가 아니면 작업 전 사용자에게 알림

---

## 프로젝트 구조

```
Lego-Digital-Twin/
├── apps/
│   ├── backend/tripo/   # FastAPI 백엔드 (카메라 → Tripo3D → Unity)
│   └── hardware/        # Arduino 제어
├── unity-client/        # Unity VR 클라이언트 (C#, XR)
├── docs/                # 환경 설정 문서
├── scripts/             # 자동화 스크립트
├── git.md               # Git 전략 (SINGLE SOURCE OF TRUTH)
└── CLAUDE.md            # 이 파일
```

## 기술 스택

- **Backend**: Python 3.9, FastAPI, Tripo3D API
- **Unity**: Unity 6000.x, C#, Meta XR SDK
- **Hardware**: Arduino, C++
- **환경**: WSL2 (Ubuntu 24.04), CUDA 11.8

## 담당 브랜치

| 역할 | 브랜치 |
|---|---|
| Vision Engine | `feature/vision` |
| Unity 클라이언트 | `feature/unity` |
| 백엔드 | `feature/backend` |
| 하드웨어 | `feature/hardware` |

---

## 코드 작성 원칙

- Python: PEP 8, type hint 사용, 비동기(async/await) 우선
- C#: Unity 네이밍 컨벤션 준수 (PascalCase 메서드, camelCase 필드)
- 주석은 WHY가 명확하지 않을 때만 작성
- `.env` 파일은 절대 커밋하지 않음
