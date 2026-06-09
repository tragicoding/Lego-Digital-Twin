#!/usr/bin/env python3
"""
update_network_ip.py — 노트북 네트워크(와이파이) 변경 시 IP 일괄 갱신 스크립트

언제 쓰나:
  학교/집 등 와이파이가 바뀌어 Windows의 IP(ipconfig 기준)가 바뀌었을 때.
  history/network_situation.txt 의 "3-1 ~ DB URL 일괄 업데이트" 단계를 자동화한다.

사용법:
  conda activate triposr
  python scripts/update_network_ip.py <새 Windows IP>

  (주의: `conda run -n triposr python ...` 형태로 실행하면 y/N 확인 입력이
   가로채여 EOFError가 발생한다. 반드시 `conda activate` 후 실행할 것.)

하는 일:
  1. apps/frontend/.env.local 의 VITE_API_URL 을 새 IP로 갱신
  2. apps/backend/.env 의 BACKEND_HOST 를 새 IP로 갱신
  3. unity-client/.../server_config.json 의 base_url 을 새 IP로 갱신
  4. DB(assets 테이블)의 model_url / thumbnail_url 에 저장된 구 호스트
     (변경 전 VITE_API_URL / BACKEND_HOST / localhost)를 새 IP로 일괄 치환

주의:
  - WSL2 내부 IP(hostname -I) / PowerShell 포트포워딩 / 방화벽 설정은
    이 스크립트가 다루지 않는다 (Windows 쪽 작업이라 수동으로 진행).
  - 서버(.env)를 재시작해야 새 설정이 반영된다.
"""
import re
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ENV = ROOT / "apps" / "frontend" / ".env.local"
BACKEND_ENV = ROOT / "apps" / "backend" / ".env"
UNITY_SERVER_CONFIG = (
    ROOT
    / "unity-client"
    / "AURA_Lego"
    / "Assets"
    / "StreamingAssets"
    / "Config"
    / "server_config.json"
)

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def update_env_var(path: Path, key: str, new_ip: str) -> str | None:
    """env 파일의 'KEY=http://host:port' 줄을 새 IP로 치환하고, 기존 host:port를 반환한다."""
    text = path.read_text()
    m = re.search(rf"^{key}=http://([^:\s]+):?(\d*)\s*$", text, re.MULTILINE)
    if not m:
        print(f"  [경고] {path.relative_to(ROOT)} 에서 {key} 줄을 찾지 못함 — 건너뜀")
        return None

    old_host, old_port = m.group(1), (m.group(2) or "8000")
    new_line = f"{key}=http://{new_ip}:{old_port}"
    text = re.sub(rf"^{key}=http://[^\s]+\s*$", new_line, text, flags=re.MULTILINE)
    path.write_text(text)
    print(f"  {path.relative_to(ROOT)}: {key} → {new_line}")
    return f"{old_host}:{old_port}"


def update_db_urls(old_hosts: set[str], new_ip: str, port: str = "8000") -> None:
    import psycopg2

    backend_env_text = BACKEND_ENV.read_text()
    m = re.search(r"^DATABASE_URL=(.+)$", backend_env_text, re.MULTILINE)
    if not m:
        print("  [경고] DATABASE_URL을 찾지 못해 DB 업데이트를 건너뜀")
        return
    db_url = m.group(1).strip()

    new_host = f"{new_ip}:{port}"
    candidates = {h for h in old_hosts if h and h != new_host} | {f"localhost:{port}"}
    candidates.discard(new_host)

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    total = 0
    try:
        for old_host in sorted(candidates):
            old_url, new_url = f"http://{old_host}", f"http://{new_host}"
            for col in ("model_url", "thumbnail_url"):
                cur.execute(
                    f"UPDATE assets SET {col} = REPLACE({col}, %s, %s) WHERE {col} LIKE %s",
                    (old_url, new_url, f"%{old_url}%"),
                )
                if cur.rowcount:
                    print(f"  DB: {col} 의 '{old_url}' → '{new_url}' ({cur.rowcount}건)")
                    total += cur.rowcount
        conn.commit()
    finally:
        cur.close()
        conn.close()
    print(f"  DB 총 {total}건 변경")


def update_unity_server_config(path: Path, new_ip: str, port: str = "8000") -> str | None:
    """Unity server_config.json의 base_url을 새 IP로 갱신하고 기존 host:port를 반환한다."""
    if not path.exists():
        print(f"  [경고] {path.relative_to(ROOT)} 파일이 없어 Unity 설정 갱신을 건너뜀")
        return None

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  [경고] {path.relative_to(ROOT)} JSON 파싱 실패 — Unity 설정 갱신을 건너뜀")
        return None

    old_url = data.get("base_url")
    old_host_port = None
    if isinstance(old_url, str):
        m = re.match(r"^http://([^:\s]+):?(\d*)$", old_url.strip())
        if m:
            old_host = m.group(1)
            old_port = m.group(2) or port
            old_host_port = f"{old_host}:{old_port}"

    new_url = f"http://{new_ip}:{port}"
    data["base_url"] = new_url
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"  {path.relative_to(ROOT)}: base_url → {new_url}")
    return old_host_port


def main() -> None:
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    yes_flag = "--yes" in sys.argv or "-y" in sys.argv
    if len(positional) != 1 or not IP_RE.match(positional[0].strip()):
        print(__doc__)
        sys.exit(1)
    new_ip = positional[0].strip()

    print(f"새 IP: {new_ip}\n")

    print("[1/4] frontend/.env.local 갱신")
    old_frontend = update_env_var(FRONTEND_ENV, "VITE_API_URL", new_ip)

    print("[2/4] backend/.env 갱신")
    old_backend = update_env_var(BACKEND_ENV, "BACKEND_HOST", new_ip)

    print("[3/4] Unity server_config.json 갱신")
    old_unity = update_unity_server_config(UNITY_SERVER_CONFIG, new_ip)

    old_hosts = {h for h in (old_frontend, old_backend, old_unity) if h}
    print(f"\n[4/4] DB(assets.model_url/thumbnail_url) 치환 대상 구 호스트: {sorted(old_hosts) + ['localhost:8000']}")
    if yes_flag:
        answer = "y"
        print("DB를 위 새 IP로 일괄 변경할까요? (y/N): y  [--yes 플래그로 자동 확인]")
    else:
        answer = input("DB를 위 새 IP로 일괄 변경할까요? (y/N): ").strip().lower()
    if answer == "y":
        update_db_urls(old_hosts, new_ip)
    else:
        print("  DB 변경을 건너뜀")

    print("\n완료. FastAPI / RQ Worker / Frontend(dev server)를 재시작해야 새 설정이 반영됩니다.")
    print("Unity는 Windows 쪽 작업본이 WSL 변경분을 pull 한 뒤 다시 열어야 새 base_url을 읽습니다.")
    print("(WSL2 포트포워딩·방화벽 설정은 이 스크립트가 다루지 않으니 history/network_situation.txt 참고)")


if __name__ == "__main__":
    main()
