#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/jskim/miniconda3/envs/triposr/bin/python"

WINDOWS_IP="${1:-}"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "[error] required command not found: $name" >&2
    exit 1
  fi
}

validate_ipv4() {
  local ip="$1"
  [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

detect_wsl_ip() {
  hostname -I | awk '{print $1}'
}

run_docker_compose() {
  echo "[2/4] Starting Docker services (db, redis)"
  (
    cd "$ROOT_DIR"
    docker compose up -d db redis
  )
}

prompt_windows_ip() {
  if [[ -n "$WINDOWS_IP" ]]; then
    return
  fi

  read -r -p "Windows Wi-Fi IP를 입력하세요: " WINDOWS_IP
}

update_project_network_ip() {
  echo "[1/4] Updating project network config with Windows IP: $WINDOWS_IP"
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" scripts/update_network_ip.py "$WINDOWS_IP" --yes
  )
}

kill_listening_port() {
  local port="$1"
  local name="$2"
  local pids

  pids="$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"
  if [[ -z "$pids" ]]; then
    echo "[$name] no existing listener on :$port"
    return
  fi

  echo "[$name] stopping existing listener on :$port (pid: $pids)"
  kill $pids 2>/dev/null || true

  for _ in {1..20}; do
    if ! ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port"; then
      return
    fi
    sleep 0.2
  done

  echo "[$name] force stopping stubborn listener on :$port"
  kill -9 $pids 2>/dev/null || true
}

stop_worker() {
  local pids
  pids="$(pgrep -f "$PYTHON_BIN run_worker.py" || true)"
  if [[ -z "$pids" ]]; then
    echo "[worker] no existing worker"
    return
  fi

  echo "[worker] stopping existing worker (pid: $pids)"
  kill $pids 2>/dev/null || true
}

clear_frontend_cache() {
  local cache_dir="$ROOT_DIR/apps/frontend/node_modules/.vite"

  if [[ -d "$cache_dir" ]]; then
    echo "[frontend] clearing Vite optimized dependency cache"
    rm -rf -- "$cache_dir"
  else
    echo "[frontend] Vite optimized dependency cache already clean"
  fi
}

prepare_fresh_app_stack() {
  echo "[3/4] Restarting app stack so new IP settings are applied"
  kill_listening_port "8000" "backend"
  stop_worker
  kill_listening_port "3000" "frontend"
  kill_listening_port "3001" "admin"
  clear_frontend_cache
}

start_app_stack() {
  echo "[4/4] Starting exhibition app stack"
  (
    cd "$ROOT_DIR"
    bash scripts/run_exhibition.sh
  )
}

main() {
  require_command docker
  require_command hostname
  require_command awk
  require_command ss
  require_command pgrep
  require_command wslpath

  prompt_windows_ip

  if ! validate_ipv4 "$WINDOWS_IP"; then
    echo "[error] invalid IPv4 address: $WINDOWS_IP" >&2
    exit 1
  fi

  update_project_network_ip
  run_docker_compose
  prepare_fresh_app_stack
  start_app_stack

  echo
  echo "Startup automation complete."
  echo "Windows IP : $WINDOWS_IP"
}

main "$@"
