# Git 전략 — Lego Digital Twin

> 이 파일은 팀원과 Claude 에이전트 모두가 읽고 그대로 따르는 단일 출처(Single Source of Truth)입니다.
> Claude Code는 이 파일을 자동으로 로드하여 모든 git 작업에 적용합니다.

---

## 기본 원칙

- `develop`에 직접 commit/push하지 않는다.
- 반드시 역할별 브랜치 또는 feature 브랜치에서 작업한다.
- 작업 전 현재 브랜치와 변경사항을 확인한다.
- PR 대상은 `develop`이다.
- `main`/`master`는 최종 안정 버전 또는 배포 버전으로만 사용한다.
- AI Agent는 사용자의 승인 없이 commit/push하지 않는다.

---

## 브랜치 구조

```
main                    ← 최종 안정 버전 (직접 push 금지)
└── develop             ← 통합 브랜치 (모든 PR 대상)
    ├── feature/unity       ← Unity 개발 (Windows)
    ├── feature/backend     ← Backend 개발 (Linux/WSL)
    ├── feature/frontend    ← Frontend 개발 (Linux/WSL)
    ├── feature/system      ← System 자동화
    └── test_app            ← 실험·기획 검증
```

| 브랜치 | 역할 | PR 대상 |
|---|---|---|
| `main` | 배포 전용 | — |
| `develop` | 통합·검증 | `main` |
| `feature/unity` | Unity 개발 | `develop` |
| `feature/backend` | Backend 개발 | `develop` |
| `feature/frontend` | Frontend 개발 | `develop` |
| `feature/system` | System 자동화 | `develop` |
| `test_app` | 실험·기획 검증 | `develop` |

---

## 작업 시작 절차

```bash
# 공통
git status
git branch
git fetch origin
git checkout develop
git pull origin develop

# Unity 개발자
git checkout feature/unity
git pull origin feature/unity

# Backend 개발자
git checkout feature/backend

# Frontend 개발자
git checkout feature/frontend
```

브랜치가 없으면 사용자에게 생성 여부를 먼저 확인한다.

---

## 워크플로우 규칙 (에이전트 적용 기준)

### RULE-01: 작업 시작 전 동기화 필수

```bash
git fetch origin develop
git merge origin/develop
```

충돌 발생 시 담당자에게 알리고 해결 후 진행한다.

### RULE-02: 커밋 메시지 형식

```
<type>(<scope>): <subject>
```

| type | 의미 |
|---|---|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 리팩토링 |
| `docs` | 문서 수정 |
| `chore` | 빌드·설정·패키지 변경 |
| `test` | 테스트 추가·수정 |
| `style` | 포맷·공백 등 코드 스타일 |

scope는 담당 모듈명 사용 (`backend`, `unity`, `frontend`, `system` 등).

**올바른 예시:**
```
feat(backend): Tripo3D GLB 업로드 엔드포인트 추가
fix(unity): 모델 로더 null 참조 예외 처리
docs(project): 브랜치 전략 문서화
chore(repo): storage gitignore 추가
```

**금지 패턴:**
- scope 없는 메시지 (`feat: something`)
- 내용 불명확한 메시지 (`update`, `fix bug`, `수정`)

### RULE-03: 스테이징·커밋 원칙

- `git add .` 또는 `git add -A` **금지** — 파일을 명시적으로 지정한다.
- `.env`, 시크릿, 바이너리 대용량 파일은 커밋하지 않는다.
- 하나의 커밋은 하나의 논리적 단위만 포함한다.

### RULE-04: push 대상

- `main`에 직접 push **금지**
- `develop`에 직접 push **금지**
- 항상 본인의 feature 브랜치에 push한다

### RULE-05: PR 규칙

- PR 대상 브랜치: **`develop`** (절대 `main`으로 직접 PR 금지)
- PR 제목: 커밋 메시지 형식과 동일
- PR 본문: 변경 내용, 테스트 방법, 영향 범위 포함

### RULE-06: 금지 명령

다음 명령은 명시적 사용자 승인 없이 절대 실행하지 않는다:

```
git push --force / git push -f
git reset --hard
git rebase -i
git commit --amend (이미 push된 커밋)
git checkout -- .
git clean -f
```

---

## 커밋 전 체크리스트

- [ ] `git status` 확인
- [ ] 변경 파일 목록 확인
- [ ] 의도하지 않은 파일 포함 여부 확인
- [ ] `.env` 포함 여부 확인
- [ ] Unity `Library/Temp/obj/Logs` 포함 여부 확인
- [ ] 관련 README 업데이트 여부 확인

---

## Unity 전용 규칙

- Unity 파일 이동은 반드시 **Unity Editor 내에서** 진행한다.
- `.meta` 파일을 직접 삭제·이동하지 않는다.
- `Library/`, `Temp/`, `obj/`, `Logs/` 커밋 금지.
- `ProjectSettings/`, `Packages/` 변경은 담당자와 협의 후 진행.
- `.unity`, `.prefab`, `.asset` 충돌은 Unity 담당자와 협의 후 해결.

---

## 충돌 해결 절차

1. `git merge origin/develop` 실행
2. 충돌 파일 목록 확인 (`git status`)
3. 각 파일을 열어 `<<<<<<<` 마커를 직접 해결
4. 해결 후 `git add <파일>` → `git commit`
5. Unity `.unity` / `.asset` 파일 충돌은 **반드시 Unity 담당자와 협의** 후 처리

---

## AI Agent 자동화 규칙 (AGENT-RULES)

Claude 에이전트는 git 관련 작업 시 아래 규칙을 자동 적용한다:

```
AGENT-GIT-01: 커밋 전 반드시 origin/develop fetch·merge 여부를 확인한다.
AGENT-GIT-02: 커밋 메시지는 반드시 <type>(<scope>): <subject> 형식을 따른다.
AGENT-GIT-03: git add는 파일을 명시적으로 지정한다. add . 금지.
AGENT-GIT-04: push는 feature/* 또는 test_app 브랜치에만 한다. main·develop 직접 push 금지.
AGENT-GIT-05: force push, reset --hard 등 파괴적 명령은 사용자 확인 후 실행한다.
AGENT-GIT-06: PR은 develop을 대상으로 생성한다.
AGENT-GIT-07: 현재 브랜치가 feature/* 또는 test_app이 아니면 작업 전에 사용자에게 알린다.
AGENT-GIT-08: 사용자의 명시적 승인 없이 commit/push하지 않는다.
AGENT-GIT-09: 브랜치 생성·삭제·merge·rebase 전 반드시 사용자에게 확인한다.
```
