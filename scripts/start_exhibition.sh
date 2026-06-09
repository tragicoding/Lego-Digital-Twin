#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/jskim/anaconda3/envs/triposr/bin/python"

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
  echo "[2/3] Starting Docker services (db, redis)"
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
  echo "[1/3] Updating project network config with Windows IP: $WINDOWS_IP"
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" scripts/update_network_ip.py "$WINDOWS_IP"
  )
}

start_app_stack() {
  echo "[3/3] Starting exhibition app stack"
  (
    cd "$ROOT_DIR"
    bash scripts/run_exhibition.sh
  )
}

main() {
  require_command docker
  require_command hostname
  require_command awk
  require_command wslpath

  prompt_windows_ip

  if ! validate_ipv4 "$WINDOWS_IP"; then
    echo "[error] invalid IPv4 address: $WINDOWS_IP" >&2
    exit 1
  fi

  update_project_network_ip
  run_docker_compose
  start_app_stack

  echo
  echo "Startup automation complete."
  echo "Windows IP : $WINDOWS_IP"
}

main "$@"
