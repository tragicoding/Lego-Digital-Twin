# Git 전략 — Lego Digital Twin

> 이 파일은 팀원과 Claude 에이전트 모두가 읽고 그대로 따르는 단일 출처(Single Source of Truth)입니다.
> Claude Code는 이 파일을 자동으로 로드하여 모든 git 작업에 적용합니다.

---

## 브랜치 구조

```
main                  ← 최종 배포 (직접 커밋·push 금지)
└── develop           ← 통합 브랜치 (모든 PR의 대상)
    ├── feature/unity     ← Unity 클라이언트
    ├── feature/vision    ← Vision Engine
    ├── feature/backend   ← 백엔드 서버
    ├── feature/hardware  ← 하드웨어 제어
    └── feature/test      ← 통합 테스트
```

| 브랜치 | 역할 | PR 대상 |
|---|---|---|
| `main` | 배포 전용 | — |
| `develop` | 통합·검증 | `main` |
| `feature/*` | 기능 개발 | `develop` |

---

## 담당 브랜치

| 역할 | 브랜치 |
|---|---|
| Vision Engine | `feature/vision` |
| Unity 클라이언트 | `feature/unity` |
| 백엔드 | `feature/backend` |
| 하드웨어 | `feature/hardware` |

---

## 워크플로우 규칙 (에이전트 적용 기준)

### RULE-01: 작업 시작 전 동기화 필수

```bash
git fetch origin develop
git merge origin/develop
```

- 작업 시작 시 **항상** `origin/develop`을 fetch·merge한다.
- 충돌이 발생하면 작업자에게 알리고 해결 후 진행한다.

### RULE-02: 커밋 메시지 형식

```
<type>(<scope>): <subject>
```

| type | 의미 |
|---|---|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 리팩토링 (기능 변경 없음) |
| `docs` | 문서 수정 |
| `chore` | 빌드·설정·패키지 변경 |
| `test` | 테스트 추가·수정 |
| `style` | 포맷·공백 등 코드 스타일 |

scope는 담당 모듈명을 사용한다 (`backend`, `unity`, `vision`, `hardware`, `map` 등).

**올바른 예시:**
```
feat(backend): Tripo3D GLB 업로드 엔드포인트 추가
fix(unity): 모델 로더 null 참조 예외 처리
refactor(vision): 이미지 전처리 파이프라인 분리
docs(project): 브랜치 전략 문서화
chore(setting): Android 빌드 플랫폼 전환
```

**금지 패턴:**
- `update`, `fix bug`, `작업`, `수정` 등 내용 불명확한 메시지
- scope 없는 메시지 (`feat: something`)
- 영문·한글 혼용 subject (둘 중 하나로 통일)

### RULE-03: 스테이징·커밋 원칙

- `git add .` 또는 `git add -A` **금지** — 파일을 명시적으로 지정한다.
- `.env`, 시크릿, 바이너리 대용량 파일은 커밋하지 않는다.
- 하나의 커밋은 하나의 논리적 단위만 포함한다.

### RULE-04: push 대상

```bash
git push origin feature/<본인브랜치>
```

- `main`에 직접 push **금지**.
- `develop`에 직접 push **금지**.
- 항상 본인의 `feature/*` 브랜치에 push한다.

### RULE-05: PR 규칙

- PR 대상 브랜치: **`develop`** (절대 `main`으로 직접 PR 금지)
- PR 제목: 커밋 메시지 형식과 동일 (`feat(backend): ...`)
- PR 본문에는 변경 내용, 테스트 방법, 스크린샷(UI 변경 시)을 포함한다.
- 병합 후 feature 브랜치는 삭제하지 않는다 (이력 보존).

### RULE-06: 금지 명령

다음 명령은 명시적 사용자 승인 없이 절대 실행하지 않는다:

```
git push --force / git push -f
git reset --hard
git rebase -i (인터랙티브)
git commit --amend (이미 push된 커밋)
git checkout -- .
git clean -f
```

---

## 표준 워크플로우 요약

```bash
# 1. 작업 시작
git checkout feature/<본인브랜치>
git fetch origin develop
git merge origin/develop

# 2. 작업 후 커밋
git add <파일 목록>
git commit -m "feat(scope): 작업 내용"

# 3. push
git push origin feature/<본인브랜치>

# 4. GitHub에서 feature/* → develop PR 생성
```

---

## 충돌 해결 절차

1. `git merge origin/develop` 실행
2. 충돌 파일 목록 확인 (`git status`)
3. 각 파일을 열어 `<<<<<<<` 마커를 직접 해결
4. 해결 후 `git add <파일>` → `git commit`
5. Unity `.unity` / `.asset` 파일 충돌은 **반드시 Unity 담당자와 협의** 후 처리

---

## 에이전트 자동화 규칙 요약 (AGENT-RULES)

Claude 에이전트는 git 관련 작업 시 아래 규칙을 자동 적용한다:

```
AGENT-GIT-01: 커밋 전 반드시 origin/develop fetch·merge 여부를 확인한다.
AGENT-GIT-02: 커밋 메시지는 반드시 <type>(<scope>): <subject> 형식을 따른다.
AGENT-GIT-03: git add는 파일을 명시적으로 지정한다. add . 금지.
AGENT-GIT-04: push는 feature/* 브랜치에만 한다. main·develop 직접 push 금지.
AGENT-GIT-05: force push, reset --hard 등 파괴적 명령은 사용자 확인 후 실행한다.
AGENT-GIT-06: PR은 develop을 대상으로 생성한다.
AGENT-GIT-07: 현재 브랜치가 feature/*가 아니면 작업 전에 사용자에게 알린다.
```
