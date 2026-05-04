import time
import os
import subprocess
import glob
import shutil
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

PYTHON       = sys.executable
BASE_DIR     = Path(__file__).parent
LOG_FILE     = BASE_DIR / "benchmark_log.json"
CONFIG_PATH  = BASE_DIR / "instant-nsr-pl/configs/neuralangelo-ortho-wmask.yaml"
WEIGHTS_PATH = BASE_DIR / "weights/RealESRGAN_x4plus.pth"


# ── 벤치마크 유틸 ──────────────────────────────────────────────────────────────

def _get_gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")[0]
        name, mem_total, mem_used, temp, util = [s.strip() for s in out.split(",")]
        return {"gpu_name": name, "mem_total_mb": int(mem_total),
                "mem_used_mb": int(mem_used), "temperature": int(temp), "utilization": int(util)}
    except Exception:
        return {}


def _read_config_params() -> dict:
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        geo = cfg["model"]["geometry"]
        enc = geo["xyz_encoding_config"]
        return {
            "resolution":           geo["isosurface"]["resolution"],
            "n_levels":             enc["n_levels"],
            "n_features_per_level": enc["n_features_per_level"],
            "hashmap_size":         enc["log2_hashmap_size"],
            "base_resolution":      enc["base_resolution"],
            "train_num_rays":       cfg["model"]["train_num_rays"],
            "max_train_num_rays":   cfg["model"]["max_train_num_rays"],
            "num_samples_per_ray":  cfg["model"]["num_samples_per_ray"],
            "ray_chunk":            cfg["model"]["ray_chunk"],
            "max_steps":            cfg["trainer"]["max_steps"],
            "precision":            cfg["trainer"]["precision"],
            "lr_geometry":          cfg["system"]["optimizer"]["params"]["geometry"]["lr"],
            "lr_texture":           cfg["system"]["optimizer"]["params"]["texture"]["lr"],
            "mlp_neurons":          geo["mlp_network_config"]["n_neurons"],
            "mlp_hidden_layers":    geo["mlp_network_config"]["n_hidden_layers"],
            "note":                 "auto_pipeline",
        }
    except Exception:
        return {"note": "auto_pipeline"}


def _mesh_stats(scene_name: str) -> dict:
    pattern = str(BASE_DIR / "instant-nsr-pl/exp" / f"{scene_name}@*" / "save" / "*.obj")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        return {"vertices": 0, "faces": 0, "obj_path": None}
    obj_path = candidates[-1]
    verts = faces = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts += 1
            elif line.startswith("f "):
                faces += 1
    return {"vertices": verts, "faces": faces, "obj_path": obj_path}


def _save_record(record: dict):
    records = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            records = json.load(f)
    records.append(record)
    with open(LOG_FILE, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[BENCHMARK] 기록 저장 완료 → {LOG_FILE} (총 {len(records)}회)")


def _refresh_dashboard():
    dashboard_script = BASE_DIR / "benchmark_dashboard.py"
    if dashboard_script.exists():
        subprocess.run([PYTHON, str(dashboard_script)], capture_output=True)
        print(f"[BENCHMARK] 대시보드 갱신 완료 → benchmark_dashboard.html")


# ── Real-ESRGAN 업스케일 ───────────────────────────────────────────────────────

def _upscale_scene_images(scene_dir: str):
    """Wonder3D가 뽑은 256×256 6장 이미지를 Real-ESRGAN으로 1024×1024로 업스케일."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(WEIGHTS_PATH),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )

    targets = [f for f in os.listdir(scene_dir)
               if (f.startswith("rgb_000_") or f.startswith("normals_000_"))
               and f.endswith(".png")]

    print(f"[ESRGAN] {len(targets)}장 업스케일 중 (256→1024)...")
    for fname in targets:
        fpath = os.path.join(scene_dir, fname)
        img   = Image.open(fpath).convert("RGBA")
        rgb   = np.array(img.convert("RGB"))
        alpha = np.array(img.split()[3])

        out_rgb, _ = upsampler.enhance(rgb, outscale=4)

        alpha_up = np.array(
            Image.fromarray(alpha).resize((out_rgb.shape[1], out_rgb.shape[0]),
                                          Image.LANCZOS)
        )
        out_rgba = np.dstack([out_rgb, alpha_up]).astype(np.uint8)
        Image.fromarray(out_rgba).save(fpath)

    print(f"[ESRGAN] 업스케일 완료 → 각 이미지 1024×1024")
    import torch
    torch.cuda.empty_cache()


# ── 파이프라인 ─────────────────────────────────────────────────────────────────

class LegoImageHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing = False

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.png'):
            if self.processing:
                return
            print(f"\n[TRIGGER] 새로운 레고 사진 감지: {event.src_path}")
            self.run_pipeline(event.src_path)

    def run_pipeline(self, image_path):
        self.processing = True
        total_start = time.time()
        step1_elapsed = step2_elapsed = 0
        scene_name = None
        success = False

        gpu_before = _get_gpu_info()
        params = _read_config_params()

        try:
            # ================================================================
            # [STEP 1] 2D 도면 및 노멀맵 생성
            # ================================================================
            print(f"\n>>> [STEP 1] 6방향 도면 생성 시작: {image_path}")
            t1 = time.time()
            subprocess.run([PYTHON, "headless_mv.py", image_path], check=True)
            step1_elapsed = time.time() - t1
            print(f"[STEP 1 완료] {step1_elapsed:.1f}초")

            # ================================================================
            # [STEP 2] 3D 폴리곤 재구성
            # ================================================================
            print("\n>>> [STEP 2] 3D 메쉬(.obj) 재구성 가동 (instant-nsr-pl)...")
            print("[INFO] 그래픽카드가 찰흙을 깎기 시작합니다. (약 1~2분 소요)")

            base_out_dir = "outputs/cropsize-192-cfg1.0"
            folders = glob.glob(os.path.join(base_out_dir, "scene@*"))
            if not folders:
                raise Exception("생성된 도면 폴더를 찾을 수 없습니다!")

            latest_scene = max(folders, key=os.path.getctime)
            scene_name = os.path.basename(latest_scene)
            print(f"[*] 타겟 씬(Scene) 감지: {scene_name}")

            # ── Bridge Code: AI 배경 제거 및 Alpha 동기화 ──────────────────
            print("[*] 3D 조각가를 위해 데이터 규격을 맞춥니다 (Rembg Alpha Sync)...")
            from rembg import remove

            normals_dir = os.path.join(latest_scene, "normals")
            masked_colors_dir = os.path.join(latest_scene, "masked_colors")

            if os.path.exists(normals_dir):
                for file_name in os.listdir(normals_dir):
                    src = os.path.join(normals_dir, file_name)
                    new_name = file_name if file_name.startswith("normals_") else f"normals_{file_name}"
                    dst = os.path.join(latest_scene, new_name)
                    shutil.move(src, dst)

            if not os.path.exists(masked_colors_dir):
                os.makedirs(masked_colors_dir)

            for img_name in os.listdir(latest_scene):
                if img_name.startswith("rgb_000_") and img_name.endswith(".png"):
                    dir_name = img_name.replace("rgb_000_", "")
                    rgb_path = os.path.join(latest_scene, img_name)
                    normal_path = os.path.join(latest_scene, f"normals_000_{dir_name}")
                    masked_color_path = os.path.join(masked_colors_dir, img_name)

                    rgb_img = Image.open(rgb_path)
                    rgba_img = remove(rgb_img)
                    rgba_img.save(masked_color_path)

                    if os.path.exists(normal_path):
                        _, _, _, alpha = rgba_img.split()
                        normal_img = Image.open(normal_path).convert("RGB")
                        r, g, b = normal_img.split()
                        rgba_normal = Image.merge("RGBA", (r, g, b, alpha))
                        rgba_normal.save(normal_path)

            print("[*] 투명도(Alpha) 및 폴더 구조 완벽 동기화 완료!")
            # ──────────────────────────────────────────────────────────────

            # ================================================================
            # [STEP 1.5] Real-ESRGAN: 멀티뷰 이미지 256→1024 업스케일
            # ================================================================
            print("\n>>> [STEP 1.5] Real-ESRGAN 텍스처 업스케일 (256×256 → 1024×1024)")
            _upscale_scene_images(latest_scene)

            recon_cmd = (
                f"cd instant-nsr-pl && "
                f"{PYTHON} launch.py --config configs/neuralangelo-ortho-wmask.yaml --gpu 0 "
                f"--train dataset.root_dir=../outputs/cropsize-192-cfg1.0/ dataset.scene={scene_name} "
                f"trainer.num_sanity_val_steps=0 dataset.num_workers=0 "
                f"model.train_num_rays=128 model.max_train_num_rays=128"
            )
            t2 = time.time()
            subprocess.run(recon_cmd, shell=True, check=True)
            step2_elapsed = time.time() - t2
            print(f"[STEP 2 완료] {step2_elapsed:.1f}초")

            success = True

            # ================================================================
            # [완료] 결과 보고
            # ================================================================
            total_elapsed = time.time() - total_start
            print("\n" + "="*50)
            print(f" [SUCCESS] 디지털 트윈 자동화 파이프라인 관통 완료!")
            print(f" - STEP 1   (멀티뷰 생성)  : {step1_elapsed:.1f}초")
            print(f" - STEP 1.5 (ESRGAN 업스케일): 포함")
            print(f" - STEP 2   (3D 재구성)    : {step2_elapsed:.1f}초")
            print(f" - 총 소요 시간            : {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")
            print(f" - 최종 3D 파일 : instant-nsr-pl/exp/{scene_name}/.../save/ 내의 .obj 파일")
            print("="*50 + "\n")

        except Exception as e:
            print(f"\n[ERROR] 파이프라인 가동 중 치명적 오류 발생: {e}")

        finally:
            total_elapsed = time.time() - total_start
            gpu_after = _get_gpu_info()
            mesh = _mesh_stats(scene_name) if scene_name else {}

            record = {
                "timestamp":   datetime.now().isoformat(timespec="seconds"),
                "scene":       scene_name or "unknown",
                "input_image": image_path,
                "params":      params,
                "elapsed_sec": round(total_elapsed, 1),
                "elapsed_min": round(total_elapsed / 60, 2),
                "step1_sec":   round(step1_elapsed, 1),
                "step2_sec":   round(step2_elapsed, 1),
                "success":     success,
                "mesh":        {k: v for k, v in mesh.items() if k != "obj_path"},
                "obj_path":    mesh.get("obj_path"),
                "gpu_before":  gpu_before,
                "gpu_after":   gpu_after,
            }
            _save_record(record)
            _refresh_dashboard()

            self.processing = False


if __name__ == "__main__":
    path = "./inputs"
    if not os.path.exists(path):
        os.makedirs(path)

    event_handler = LegoImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)

    print("="*50)
    print("  Lego-Digital-Twin 자동 감시 모니터 가동 ")
    print(f" - 대상 폴더: {os.path.abspath(path)}")
    print(" - 상태: 사진이 저장되면 자동으로 3D 모델링을 시작합니다.")
    print(" - 벤치마크: 매 실험 결과가 benchmark_log.json에 자동 기록됩니다.")
    print("="*50)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
