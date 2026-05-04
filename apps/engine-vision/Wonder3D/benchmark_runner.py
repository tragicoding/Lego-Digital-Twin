"""
benchmark_runner.py
파라미터 조합 1개를 실행하고 결과를 benchmark_log.json에 기록한다.

사용법:
  python benchmark_runner.py --scene <scene_name> \
    --resolution 256 --n_levels 12 --max_steps 3000 \
    --train_num_rays 128 --hashmap_size 19 --note "테스트1"

scene_name 예: scene@20260503-141351
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR    = Path(__file__).parent
CONFIG_SRC  = BASE_DIR / "instant-nsr-pl/configs/neuralangelo-ortho-wmask.yaml"
LOG_FILE    = BASE_DIR / "benchmark_log.json"
OUTPUTS_DIR = BASE_DIR / "outputs/cropsize-192-cfg1.0"
EXP_DIR     = BASE_DIR / "instant-nsr-pl/exp"
PYTHON      = sys.executable


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")[0]
        name, mem_total, mem_used, temp, util = [s.strip() for s in out.split(",")]
        return {
            "gpu_name":   name,
            "mem_total_mb": int(mem_total),
            "mem_used_mb":  int(mem_used),
            "temperature":  int(temp),
            "utilization":  int(util),
        }
    except Exception:
        return {}


def mesh_stats(obj_path: Path) -> dict:
    if not obj_path.exists():
        return {"vertices": 0, "faces": 0}
    verts = faces = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts += 1
            elif line.startswith("f "):
                faces += 1
    return {"vertices": verts, "faces": faces}


def find_latest_obj(scene_name: str) -> Path | None:
    pattern = EXP_DIR / f"{scene_name}@*" / "save" / "*.obj"
    import glob
    candidates = sorted(glob.glob(str(pattern)), key=os.path.getmtime)
    return Path(candidates[-1]) if candidates else None


def load_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_log(records: list):
    with open(LOG_FILE, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[LOG] 저장: {LOG_FILE}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def run(args):
    # 1. config YAML 수정 (임시 파일)
    with open(CONFIG_SRC) as f:
        cfg = yaml.safe_load(f)

    cfg["geometry"]["isosurface"]["resolution"]         = args.resolution
    cfg["geometry"]["xyz_encoding_config"]["n_levels"]  = args.n_levels
    cfg["geometry"]["xyz_encoding_config"]["log2_hashmap_size"] = args.hashmap_size
    cfg["model"]["train_num_rays"]                      = args.train_num_rays
    cfg["model"]["max_train_num_rays"]                  = args.train_num_rays * 2
    cfg["trainer"]["max_steps"]                         = args.max_steps

    tmp_config = BASE_DIR / "instant-nsr-pl/configs/_benchmark_tmp.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    params = {
        "resolution":     args.resolution,
        "n_levels":       args.n_levels,
        "hashmap_size":   args.hashmap_size,
        "train_num_rays": args.train_num_rays,
        "max_steps":      args.max_steps,
        "note":           args.note,
    }

    print("\n" + "="*60)
    print(f"[BENCHMARK] 파라미터: {params}")
    print("="*60)

    gpu_before = get_gpu_info()

    # 2. 재구성 실행 + 시간 측정
    cmd = (
        f"cd {BASE_DIR}/instant-nsr-pl && "
        f"{PYTHON} launch.py "
        f"--config configs/_benchmark_tmp.yaml "
        f"--gpu 0 --train "
        f"dataset.root_dir={OUTPUTS_DIR}/ "
        f"dataset.scene={args.scene} "
        f"trainer.num_sanity_val_steps=0 "
        f"dataset.num_workers=0"
    )

    t_start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t_start

    gpu_after = get_gpu_info()
    obj_path  = find_latest_obj(args.scene)
    stats     = mesh_stats(obj_path) if obj_path else {}

    success = result.returncode == 0

    record = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "scene":       args.scene,
        "params":      params,
        "elapsed_sec": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 2),
        "success":     success,
        "mesh":        stats,
        "obj_path":    str(obj_path) if obj_path else None,
        "gpu_before":  gpu_before,
        "gpu_after":   gpu_after,
    }

    records = load_log()
    records.append(record)
    save_log(records)

    print(f"\n[결과] 소요시간: {elapsed/60:.1f}분 | 버텍스: {stats.get('vertices',0):,} | 성공: {success}")
    tmp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene",         required=True,  help="scene 폴더명 (예: scene@20260503-141351)")
    p.add_argument("--resolution",    type=int, default=192)
    p.add_argument("--n_levels",      type=int, default=10)
    p.add_argument("--hashmap_size",  type=int, default=19)
    p.add_argument("--train_num_rays",type=int, default=128)
    p.add_argument("--max_steps",     type=int, default=3000)
    p.add_argument("--note",          default="")
    run(p.parse_args())
