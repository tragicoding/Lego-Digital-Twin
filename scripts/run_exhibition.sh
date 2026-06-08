#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/history/.run_logs"
PYTHON_BIN="/home/jskim/anaconda3/envs/triposr/bin/python"

mkdir -p "$LOG_DIR"

start_port_service() {
  local name="$1"
  local port="$2"
  local command="$3"
  local log_file="$LOG_DIR/$name.log"

  if ss -ltn "sport = :$port" | grep -q ":$port"; then
    echo "[$name] already listening on :$port"
    return
  fi

  nohup bash -lc "cd '$ROOT_DIR' && $command" >"$log_file" 2>&1 &
  echo $! > "$LOG_DIR/$name.pid"
  echo "[$name] started on :$port (log: $log_file)"
}

start_worker() {
  local log_file="$LOG_DIR/worker.log"
  if pgrep -f "$PYTHON_BIN run_worker.py" >/dev/null; then
    echo "[worker] already running"
    return
  fi

  nohup bash -lc "cd '$ROOT_DIR' && $PYTHON_BIN run_worker.py" >"$log_file" 2>&1 &
  echo $! > "$LOG_DIR/worker.pid"
  echo "[worker] started (log: $log_file)"
}

start_port_service \
  "backend" \
  "8000" \
  "$PYTHON_BIN -m uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000"

start_worker

start_port_service \
  "frontend" \
  "3000" \
  "cd apps/frontend && npm run dev -- --host 0.0.0.0 --port 3000"

start_port_service \
  "admin" \
  "3001" \
  "cd apps/frontend && npm run dev:admin -- --host 0.0.0.0 --port 3001"

echo
echo "MINIVERSE exhibition stack is running."
echo "frontend : http://localhost:3000"
echo "admin    : http://localhost:3001"
echo "backend  : http://localhost:8000"
