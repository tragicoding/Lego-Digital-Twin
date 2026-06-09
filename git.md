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

- `git add .` 허용 — 단, commit 전 `git status`로 포함 파일을 반드시 확인한다.
- `.env`, 시크릿, 바이너리 대용량 파일, node_modules, dist, storage는 커밋하지 않는다.
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

### RULE-07: 파일 누락 방지 — add / commit / pull / push 전 검증

#### git add 전

```bash
# 변경 파일 전체 확인
git status

# 추적되지 않은 파일 확인 (누락 위험 — 특히 신규 .meta, 새 에셋)
git ls-files --others --exclude-standard
```

> Unity의 경우 본 파일과 `.meta`가 함께 표시되는지 반드시 확인한다.  
> `.meta` 누락 시 Unity가 새 GUID를 생성해 씬/프리팹 참조가 깨진다.

#### git add 후, commit 전

```bash
# 스테이징된 파일 목록 확인 (A=추가, M=수정, D=삭제)
git diff --cached --name-status

# 변경량 요약으로 이상 여부 확인
git diff --cached --stat
```

의도한 파일이 모두 포함되었는지, 불필요한 파일(Logs, Library 등)이 없는지 확인한다.

#### git pull 전

```bash
# 로컬 미저장 변경사항 확인 (pull 시 충돌·덮어쓰기 위험)
git status

# 변경사항이 있으면 임시 저장 후 pull
git stash
git pull origin <브랜치>
git stash pop
```

#### git push 전

```bash
# push될 커밋 목록 확인 (누락 커밋 없는지 검증)
git log origin/<브랜치>..HEAD --oneline

# 워킹 트리가 깨끗한지 확인 (미커밋 파일 없어야 함)
git status
```

### RULE-08: 브랜치 폴더 교체 절차

특정 브랜치의 디렉토리 전체를 현재 브랜치로 가져올 때 반드시 아래 순서를 따른다.

**잘못된 방법 (파일 누락 위험):**
```bash
# 체크아웃 후 파일을 일일이 명시적으로 스테이징 → .meta, 바이너리 누락 가능
git checkout origin/feature/X -- path/
git add path/file1 path/file2 ...  # ← 누락 발생
```

**올바른 방법:**
```bash
# 1. 교체할 경로 체크아웃
git checkout origin/<브랜치> -- <경로>/

# 2. 변경 내용 확인 (누락 파일 없는지 검증)
git diff --name-status origin/<브랜치> -- <경로>/

# 3. 해당 디렉토리 전체를 스테이징 (디렉토리 지정은 허용)
git add <경로>/

# 4. 커밋 전 최종 검증 — D(삭제)로 표시된 파일이 없어야 정상
git diff --name-status --cached origin/<브랜치> -- <경로>/

# 5. 커밋
git commit -m "chore(<scope>): <브랜치> 내용으로 <경로> 교체"
```

**교체 후 반드시 확인:**
```bash
# 누락 파일이 남아있으면 D(삭제)로 표시됨 → 추가 checkout 필요
git diff --name-only origin/<브랜치> -- <경로>/ | head -20
```

> **Unity 주의사항**: `.meta` 파일 누락 시 Unity가 새 GUID를 생성하여  
> 씬/프리팹의 스크립트·에셋 참조가 모두 깨진다. 반드시 함께 포함.

---

## 커밋 전 체크리스트

**add 전**
- [ ] `git status` — 변경 파일 전체 목록 파악
- [ ] `git ls-files --others --exclude-standard` — 누락 가능한 untracked 파일 확인
- [ ] Unity: 본 파일과 `.meta` 파일이 함께 표시되는지 확인

**add 후, commit 전**
- [ ] `git diff --cached --name-status` — 스테이징된 파일 목록 재확인
- [ ] `git diff --cached --stat` — 변경량 요약으로 이상 여부 확인
- [ ] 의도하지 않은 파일 포함 여부 확인 (Logs, Library, node_modules 등)
- [ ] `.env` 포함 여부 확인
- [ ] Unity `Library/Temp/obj/Logs` 포함 여부 확인
- [ ] 관련 README 업데이트 여부 확인

**push 전**
- [ ] `git log origin/<브랜치>..HEAD --oneline` — push될 커밋 목록 확인
- [ ] `git status` — 워킹 트리 깨끗한지 확인 (미커밋 파일 없어야 함)

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
AGENT-GIT-03: git add . 허용. 단 commit 전 git status로 포함 파일 확인 필수.
AGENT-GIT-04: push는 feature/* 또는 test_app 브랜치에만 한다. main·develop 직접 push 금지.
AGENT-GIT-05: force push, reset --hard 등 파괴적 명령은 사용자 확인 후 실행한다.
AGENT-GIT-06: PR은 develop을 대상으로 생성한다.
AGENT-GIT-07: 현재 브랜치가 feature/* 또는 test_app이 아니면 작업 전에 사용자에게 알린다.
AGENT-GIT-08: 사용자의 명시적 승인 없이 commit/push하지 않는다.
AGENT-GIT-09: 브랜치 생성·삭제·merge·rebase 전 반드시 사용자에게 확인한다.
AGENT-GIT-10: 브랜치 폴더 교체 후 git diff --name-status origin/<브랜치> -- <경로>/ 로
             누락 파일(D 표시)이 없는지 반드시 검증한다.
             누락 파일이 있으면 추가 checkout 후 함께 커밋한다.
AGENT-GIT-11: git add 전 git ls-files --others --exclude-standard 로 누락 가능한
             untracked 파일(특히 .meta, 새 에셋)이 없는지 확인한다.
AGENT-GIT-12: git add 후 커밋 전 git diff --cached --name-status 로 스테이징된
             파일 목록을 반드시 확인하고, 의도하지 않은 파일이 없는지 검증한다.
AGENT-GIT-13: git push 전 git log origin/<브랜치>..HEAD --oneline 으로 push될
             커밋 목록을 확인하고 누락된 커밋이 없는지 검증한다.
AGENT-GIT-14: git pull 전 git status 로 로컬 변경사항을 확인한다.
             변경사항이 있으면 commit 또는 stash 처리 후 pull한다.
```
