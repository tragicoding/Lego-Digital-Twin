#!/usr/bin/env bash
# git-workflow.sh — git.md 전략 기반 워크플로우 자동화
# 사용법: ./scripts/git-workflow.sh <command> [args]
#
# Commands:
#   sync              develop fetch·merge (RULE-01)
#   commit <msg>      규칙 검증 후 커밋 (RULE-02, RULE-03)
#   push              현재 feature/* 브랜치에 push (RULE-04)
#   start <branch>    feature 브랜치 생성 후 sync
#   status            현재 브랜치·sync 상태 요약
#   install-hooks     commit-msg git hook 설치

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

current_branch() { git rev-parse --abbrev-ref HEAD; }

assert_feature_branch() {
  local branch
  branch="$(current_branch)"
  if [[ "$branch" != feature/* ]]; then
    error "현재 브랜치가 '${branch}'입니다. feature/* 브랜치에서만 실행하세요. (RULE-04)"
  fi
}

validate_commit_msg() {
  local msg="$1"
  local pattern='^(feat|fix|refactor|docs|chore|test|style)\([a-zA-Z0-9_/-]+\): .+'
  if ! echo "$msg" | grep -qE "$pattern"; then
    echo ""
    error "커밋 메시지 형식 오류 (RULE-02)
  올바른 형식: <type>(<scope>): <subject>
  예시: feat(backend): Tripo3D 업로드 엔드포인트 추가
  허용 type: feat, fix, refactor, docs, chore, test, style
  입력값: '${msg}'"
  fi
}

cmd_sync() {
  info "origin/develop fetch·merge 중... (RULE-01)"
  git fetch origin develop
  local branch
  branch="$(current_branch)"
  if git merge origin/develop --no-edit; then
    ok "'${branch}' 브랜치가 origin/develop과 동기화되었습니다."
  else
    warn "충돌이 발생했습니다. 파일을 수동으로 해결한 후 'git add <파일> && git commit' 하세요."
    git status
    exit 1
  fi
}

cmd_commit() {
  local msg="${1:-}"
  if [[ -z "$msg" ]]; then
    error "커밋 메시지를 입력하세요.\n사용법: $0 commit \"feat(scope): 내용\""
  fi
  validate_commit_msg "$msg"

  local staged
  staged="$(git diff --cached --name-only)"
  if [[ -z "$staged" ]]; then
    warn "스테이징된 파일이 없습니다. 'git add <파일>' 후 다시 실행하세요."
    exit 1
  fi

  # .env 파일 감지
  if echo "$staged" | grep -qE '\.env($|\.)'; then
    error ".env 파일이 스테이징에 포함되어 있습니다. 커밋을 중단합니다. (RULE-03)"
  fi

  info "커밋 중: ${msg}"
  git commit -m "$msg"
  ok "커밋 완료."
}

cmd_push() {
  assert_feature_branch
  local branch
  branch="$(current_branch)"
  info "'${branch}' → origin push 중... (RULE-04)"
  git push origin "$branch"
  ok "push 완료: origin/${branch}"
}

cmd_start() {
  local name="${1:-}"
  if [[ -z "$name" ]]; then
    error "브랜치 이름을 입력하세요.\n사용법: $0 start <브랜치명>\n예시: $0 start feature/backend"
  fi
  if [[ "$name" != feature/* ]]; then
    name="feature/${name}"
  fi

  info "develop 최신화 후 '${name}' 브랜치 생성..."
  git fetch origin develop
  git checkout -b "$name" origin/develop 2>/dev/null || git checkout "$name"
  ok "'${name}' 브랜치에서 작업을 시작합니다. develop과 동기화되었습니다."
}

cmd_status() {
  local branch
  branch="$(current_branch)"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  현재 브랜치: ${branch}"

  if [[ "$branch" == feature/* ]]; then
    ok "feature 브랜치 확인 완료"
  else
    warn "feature/* 브랜치가 아닙니다!"
  fi

  git fetch origin develop --quiet 2>/dev/null || true
  local behind
  behind=$(git rev-list --count HEAD..origin/develop 2>/dev/null || echo "?")
  if [[ "$behind" == "0" ]]; then
    ok "origin/develop과 동기화됨"
  else
    warn "origin/develop보다 ${behind}개 커밋 뒤처져 있습니다. './scripts/git-workflow.sh sync' 실행 권장"
  fi

  echo ""
  git status --short
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

cmd_install_hooks() {
  local hook_dir="${REPO_ROOT}/.git/hooks"
  local commit_msg_hook="${hook_dir}/commit-msg"

  info ".git/hooks/commit-msg 설치 중..."
  cat > "$commit_msg_hook" <<'HOOK'
#!/usr/bin/env bash
# commit-msg hook: git.md RULE-02 커밋 메시지 형식 검증
MSG="$(cat "$1")"
PATTERN='^(feat|fix|refactor|docs|chore|test|style)\([a-zA-Z0-9_/-]+\): .+'

# Merge·Revert 커밋은 검증 제외
if echo "$MSG" | grep -qE '^(Merge|Revert)'; then
  exit 0
fi

if ! echo "$MSG" | grep -qE "$PATTERN"; then
  echo ""
  echo "[commit-msg hook] 커밋 메시지 형식 오류 (git.md RULE-02)"
  echo "  올바른 형식: <type>(<scope>): <subject>"
  echo "  허용 type: feat, fix, refactor, docs, chore, test, style"
  echo "  예시: feat(backend): Tripo3D 업로드 엔드포인트 추가"
  echo "  입력값: '${MSG}'"
  echo ""
  exit 1
fi
HOOK

  chmod +x "$commit_msg_hook"
  ok "commit-msg hook 설치 완료: ${commit_msg_hook}"

  local prepush_hook="${hook_dir}/pre-push"
  info ".git/hooks/pre-push 설치 중..."
  cat > "$prepush_hook" <<'HOOK'
#!/usr/bin/env bash
# pre-push hook: git.md RULE-04 main·develop 직접 push 방지
REMOTE="$1"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ "$BRANCH" == "main" || "$BRANCH" == "develop" ]]; then
  echo ""
  echo "[pre-push hook] '${BRANCH}' 브랜치에 직접 push는 금지입니다. (git.md RULE-04)"
  echo "  feature/* 브랜치에서 push 후 PR을 통해 develop에 병합하세요."
  echo ""
  exit 1
fi
HOOK

  chmod +x "$prepush_hook"
  ok "pre-push hook 설치 완료: ${prepush_hook}"
  ok "모든 git hook 설치 완료. 이제 커밋 메시지 형식과 push 대상이 자동 검증됩니다."
}

# --- 진입점 ---
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
  sync)           cmd_sync ;;
  commit)         cmd_commit "$@" ;;
  push)           cmd_push ;;
  start)          cmd_start "$@" ;;
  status)         cmd_status ;;
  install-hooks)  cmd_install_hooks ;;
  help|--help|-h)
    echo "사용법: ./scripts/git-workflow.sh <command> [args]"
    echo ""
    echo "  sync              origin/develop fetch·merge (RULE-01)"
    echo "  commit <msg>      커밋 메시지 검증 후 커밋 (RULE-02, 03)"
    echo "  push              현재 feature/* 브랜치에 push (RULE-04)"
    echo "  start <name>      feature 브랜치 생성 및 sync"
    echo "  status            브랜치·sync 상태 요약"
    echo "  install-hooks     commit-msg, pre-push git hook 설치"
    ;;
  *)
    error "알 수 없는 명령: '${COMMAND}'. './scripts/git-workflow.sh help' 참조"
    ;;
esac
