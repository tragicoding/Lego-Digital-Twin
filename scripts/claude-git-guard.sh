#!/usr/bin/env bash
# claude-git-guard.sh — Claude PreToolUse 훅
# Claude가 Bash 도구로 git 명령을 실행하기 전에 git.md 규칙 위반을 차단한다.
#
# 동작: settings.json의 PreToolUse 훅에서 호출됨
# 입력: $1 = Claude가 실행하려는 bash 명령 문자열
# 종료코드: 0 = 허용, 2 = 차단 (Claude에 경고 메시지 표시)

CMD="${1:-}"

# git 명령이 아니면 통과
if ! echo "$CMD" | grep -qE '^\s*git\s+'; then
  exit 0
fi

block() {
  echo "[git-guard] 차단: $1" >&2
  echo "[git.md 규칙 위반] $2" >&2
  exit 2
}

# RULE-04: main·develop 직접 push 금지
if echo "$CMD" | grep -qE 'git\s+push\s+.*\s+(origin\s+)?(main|develop)\b'; then
  block "$CMD" "RULE-04: main·develop에 직접 push할 수 없습니다. feature/* 브랜치에 push하세요."
fi

# RULE-06: force push 금지
if echo "$CMD" | grep -qE 'git\s+push\s+(--force|-f)'; then
  block "$CMD" "RULE-06: force push는 금지입니다. 사용자에게 확인을 요청하세요."
fi

# RULE-06: reset --hard 금지
if echo "$CMD" | grep -qE 'git\s+reset\s+--hard'; then
  block "$CMD" "RULE-06: git reset --hard는 파괴적 명령입니다. 사용자 확인 없이 실행할 수 없습니다."
fi

# RULE-06: clean -f 금지
if echo "$CMD" | grep -qE 'git\s+clean\s+.*-f'; then
  block "$CMD" "RULE-06: git clean -f는 파괴적 명령입니다. 사용자 확인 없이 실행할 수 없습니다."
fi

# RULE-03: git add . / git add -A 경고 (차단하지 않고 경고만)
if echo "$CMD" | grep -qE 'git\s+add\s+(\.|--all|-A)\s*$'; then
  echo "[git-guard 경고] RULE-03: 'git add .' 또는 'git add -A' 감지. 파일을 명시적으로 지정하는 것을 권장합니다." >&2
fi

exit 0
