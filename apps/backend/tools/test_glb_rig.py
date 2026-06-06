"""
[일회성 실측 도구] Tripo animate_rig 의 GLB 출력이 스켈레톤 본 이름을
보존하는지 확인한다.

목적
  Unity 런타임 Humanoid 리타게팅(HumanoidAvatarBuilder)을 TriLib 없이
  glTFast 로 구현하려면, 백엔드를 out_format="glb" 로 바꿨을 때 리깅 GLB가
  FBX와 동일한 본 이름(Hip, L_Thigh, L_Calf, Waist, Spine01 ...)을
  보존해야 한다. 이 스크립트가 그 보존 여부를 실제로 측정한다.

실행 (백엔드 .env 가 있는 환경 = WSL/conda triposr)
  cd apps/backend
  # 방법 A) 기존 모델 task_id 재사용 (이미지→모델 비용 절약, 권장)
  python -m tools.test_glb_rig --model-task-id <TRIPO_MODEL_TASK_ID>
  # 방법 B) 이미지부터 전체 실행 (task_id 없을 때)
  python -m tools.test_glb_rig --image /path/to/character_front.png

출력
  - 다운로드한 GLB 경로
  - skins[].joints 의 본 이름 목록 (← 이게 핵심)
  - HumanoidAvatarBuilder 가 기대하는 22개 Humanoid 본과의 자동 대조표

해석
  joints 에 Hip / L_Thigh / L_Calf / Waist / Spine01 / NeckTwist01 ... 가
  보이면 → 보존됨 → No-TriLib (a) 안 진행 OK.
  joints 가 node0 / bone_3 처럼 정규화돼 있으면 → 백엔드에서 본 이름을
  명시 지정하거나 FBX(TriLib) 경로로 전환 필요.
"""
import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

# .env 로드 (apps/backend/.env)
_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV)

TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY")
TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi"
HEADERS = {"Authorization": f"Bearer {TRIPO_API_KEY}"}

# HumanoidAvatarBuilder 가 매핑해야 하는 Unity Humanoid 22본 (FBX .meta 기준 정답)
EXPECTED = {
    "Hips": "Hip", "Spine": "Waist", "Chest": "Spine01", "UpperChest": "Spine02",
    "Neck": "NeckTwist01", "Head": "Head",
    "LeftShoulder": "L_Clavicle", "RightShoulder": "R_Clavicle",
    "LeftUpperArm": "L_Upperarm", "RightUpperArm": "R_Upperarm",
    "LeftLowerArm": "L_Forearm", "RightLowerArm": "R_Forearm",
    "LeftHand": "L_Hand", "RightHand": "R_Hand",
    "LeftUpperLeg": "L_Thigh", "RightUpperLeg": "R_Thigh",
    "LeftLowerLeg": "L_Calf", "RightLowerLeg": "R_Calf",
    "LeftFoot": "L_Foot", "RightFoot": "R_Foot",
    "LeftToes": "L_ToeBase", "RightToes": "R_ToeBase",
}


async def upload_image(image_path: Path) -> str:
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    async with httpx.AsyncClient() as c:
        with open(image_path, "rb") as f:
            resp = await c.post(
                f"{TRIPO_BASE_URL}/upload/sts", headers=HEADERS,
                files={"file": (image_path.name, f, mime)}, timeout=120,
            )
        resp.raise_for_status()
        return resp.json()["data"]["image_token"]


async def create_model_task(file_token: str, ext: str) -> str:
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**HEADERS, "Content-Type": "application/json"},
            json={"type": "image_to_model",
                  "file": {"type": ext, "file_token": file_token},
                  "model_version": "v2.5-20250123"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]


async def create_rig_task_glb(model_task_id: str) -> str:
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**HEADERS, "Content-Type": "application/json"},
            json={"type": "animate_rig",
                  "original_model_task_id": model_task_id,
                  "out_format": "glb"},   # ← 핵심: GLB 로 요청
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]


async def poll_task(task_id: str, interval: float = 2.0, timeout: float = 600.0) -> dict:
    start = time.time()
    prev = None
    while time.time() - start < timeout:
        async with httpx.AsyncClient() as c:
            resp = await c.get(f"{TRIPO_BASE_URL}/task/{task_id}", headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]
        if status != prev:
            print(f"  [poll] {task_id[:8]} {prev} -> {status} (+{time.time()-start:.0f}s)", flush=True)
            prev = status
        if status.upper() in ("SUCCESS", "FINISHED"):
            return data
        if status.upper() in ("FAILED", "CANCELLED", "UNKNOWN"):
            raise RuntimeError(f"Tripo 태스크 실패: {status}")
        await asyncio.sleep(interval)
    raise TimeoutError(f"타임아웃: {task_id}")


async def download(url: str, out: Path):
    async with httpx.AsyncClient() as c:
        resp = await c.get(url, timeout=180, follow_redirects=True)
        resp.raise_for_status()
        out.write_bytes(resp.content)


def dump_glb_bones(glb_path: Path):
    """GLB 의 JSON 청크를 파싱해 노드/스킨 조인트 이름을 출력한다 (순수 파이썬)."""
    buf = glb_path.read_bytes()
    if buf[0:4] != b"glTF":
        print("  ! GLB 매직 불일치")
        return
    chunk_len = struct.unpack_from("<I", buf, 12)[0]
    gltf = json.loads(buf[20:20 + chunk_len].decode("utf-8"))
    nodes = gltf.get("nodes", [])
    skins = gltf.get("skins", [])
    anims = gltf.get("animations", [])
    print(f"\n=== GLB 구조: nodes={len(nodes)}, skins={len(skins)}, animations={len(anims)} ===")

    joint_names = []
    for si, skin in enumerate(skins):
        joints = skin.get("joints", [])
        print(f"\n--- skin[{si}] joints ({len(joints)}) ---")
        for j in joints:
            name = nodes[j].get("name", "(no name)") if j < len(nodes) else "?"
            joint_names.append(name)
            print(f"  {name}")

    if not skins:
        print("\n  ! skins 없음 — 스켈레톤 미포함 GLB (리깅 출력이 아닐 수 있음)")
        print("  전체 노드 이름:")
        for i, n in enumerate(nodes):
            print(f"    [{i}] {n.get('name', '(no name)')}")
        return

    # ── HumanoidAvatarBuilder 기대 본과 대조 ──────────────────────────
    print("\n=== Unity Humanoid 22본 보존 대조 ===")
    present = set(joint_names)
    ok = 0
    for human, expected_bone in EXPECTED.items():
        hit = expected_bone in present
        ok += hit
        print(f"  [{'O' if hit else 'X'}] {human:16s} <- {expected_bone}")
    print(f"\n  보존된 본: {ok}/22")
    if ok >= 15:
        print("  => 보존됨. No-TriLib (a) 안 진행 가능.")
    else:
        print("  => 본 이름 불일치/정규화. 위 'skin joints' 실제 이름을 확인하고")
        print("     BoneMap 을 그 이름으로 맞추거나 FBX(TriLib) 경로 검토 필요.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-task-id", help="기존 image_to_model task_id (비용 절약)")
    ap.add_argument("--image", help="--model-task-id 없을 때 사용할 입력 이미지 경로")
    ap.add_argument("--out", default="test_rigged.glb", help="저장할 GLB 경로")
    args = ap.parse_args()

    if not TRIPO_API_KEY:
        print("! TRIPO_API_KEY 없음 — apps/backend/.env 를 확인하세요.")
        sys.exit(1)

    model_task_id = args.model_task_id
    if not model_task_id:
        if not args.image:
            print("! --model-task-id 또는 --image 중 하나가 필요합니다.")
            sys.exit(1)
        img = Path(args.image)
        if not img.exists():
            print(f"! 이미지 없음: {img}")
            sys.exit(1)
        print(f"[1/3] 이미지 업로드: {img.name}")
        token = await upload_image(img)
        ext = img.suffix.lower().lstrip(".")
        ext = ext if ext in ("png", "jpg", "jpeg", "webp") else "jpg"
        print("[2/3] image_to_model 생성 중...")
        model_task_id = await create_model_task(token, ext)
        await poll_task(model_task_id)
    else:
        print(f"[*] 기존 model_task_id 재사용: {model_task_id}")

    print("[3/3] animate_rig (out_format=glb) 요청 중...")
    rig_task = await create_rig_task_glb(model_task_id)
    rig_result = await poll_task(rig_task)

    out_keys = list(rig_result.get("output", {}).keys())
    print(f"\n  rig output keys: {out_keys}")
    glb_url = rig_result["output"].get("model") or rig_result["output"].get("animation")
    if not glb_url:
        print(f"! GLB URL 없음. output: {rig_result.get('output')}")
        sys.exit(1)

    out = Path(args.out)
    await download(glb_url, out)
    print(f"\nGLB 저장: {out.resolve()} ({out.stat().st_size} bytes)")

    dump_glb_bones(out)


if __name__ == "__main__":
    asyncio.run(main())
