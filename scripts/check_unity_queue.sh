#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/jskim/miniconda3/envs/triposr/bin/python"

cd "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
import re
import sys
from pathlib import Path

from redis import Redis
from sqlalchemy import create_engine, text

env_text = Path("apps/backend/.env").read_text()

db_match = re.search(r"^DATABASE_URL=(.+)$", env_text, re.MULTILINE)
redis_match = re.search(r"^REDIS_URL=(.+)$", env_text, re.MULTILINE)

if not db_match:
    print("[error] DATABASE_URL not found in apps/backend/.env", file=sys.stderr)
    sys.exit(1)

db_url = db_match.group(1).strip()
redis_url = redis_match.group(1).strip() if redis_match else "redis://localhost:6379"

try:
    redis_conn = Redis.from_url(redis_url)
    queue = [item.decode() for item in redis_conn.lrange("lego:unity_queue", 0, -1)]
    redis_conn.close()
except Exception as exc:
    print(f"[error] failed to read redis unity_queue: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"unity_queue_length = {len(queue)}")
print(f"unity_queue = {queue}")
print()

if not queue:
    print("No queued Unity sessions.")
    sys.exit(0)

try:
    engine = create_engine(db_url)
except Exception as exc:
    print(f"[error] failed to create database engine: {exc}", file=sys.stderr)
    sys.exit(1)

session_sql = text(
    """
    select id, nickname, bubble_text, status, likes
    from sessions
    where id = :sid
    """
)

asset_sql = text(
    """
    select asset_type, status, stage, progress, model_url
    from assets
    where session_id = :sid
    order by created_at asc
    """
)

with engine.connect() as conn:
    for index, sid in enumerate(queue, start=1):
        session_row = conn.execute(session_sql, {"sid": sid}).fetchone()
        print(f"[{index}] session_id = {sid}")

        if session_row is None:
            print("    session = <missing in database>")
            print()
            continue

        print(f"    nickname    = {session_row.nickname or '-'}")
        print(f"    bubble_text = {session_row.bubble_text or '-'}")
        print(f"    status      = {session_row.status}")
        print(f"    likes       = {session_row.likes}")

        asset_rows = conn.execute(asset_sql, {"sid": sid}).fetchall()
        if not asset_rows:
            print("    assets      = <none>")
            print()
            continue

        print("    assets:")
        for asset in asset_rows:
            print(
                "      "
                f"{asset.asset_type}: "
                f"status={asset.status}, "
                f"stage={asset.stage}, "
                f"progress={asset.progress}, "
                f"model_url={asset.model_url or '-'}"
            )
        print()
PY
