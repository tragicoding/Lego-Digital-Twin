# ====================================================================
# [STEP 0] RTX 4070 & PyTorch 2.1.0 호환성 통합 핫픽스
# ====================================================================
import torch
import torch.utils._pytree
import torch.distributed

def fix_pytree(*args, **kwargs): pass
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = fix_pytree
if not hasattr(torch.utils._pytree, 'register_leaf'):
    torch.utils._pytree.register_leaf = fix_pytree

if not hasattr(torch, "xpu"):
    class DummyXPU:
        def __init__(self): self.device_count = lambda: 0; self.is_available = lambda: False
        def empty_cache(self): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = DummyXPU()

if not hasattr(torch.distributed, "device_mesh"):
    class DummyMesh: DeviceMesh = None
    torch.distributed.device_mesh = DummyMesh()
# ====================================================================

import sys
from PIL import Image
from omegaconf import OmegaConf
from utils.misc import load_config
# 기존 UI 파일에서 핵심 AI 로직만 그대로 가져옵니다.
from gradio_app_mv import load_wonder3d_pipeline, sam_init, preprocess, run_pipeline, TestConfig

def main(image_path):
    print(f"\n[Headless] 백그라운드 작업 시작: {image_path}")

    # 1. 설정 로드
    cfg = load_config("./configs/mvdiffusion-joint-ortho-6views.yaml")
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    # 2. 엔진 로드 (UI 없이 메모리에 바로 올림)
    pipeline = load_wonder3d_pipeline(cfg)
    pipeline.to('cuda:0')
    predictor = sam_init()

    # 3. 전처리 (배경 투명화 및 리사이즈)
    print("[Headless] 이미지 분석 및 배경 제거 중...")
    input_image = Image.open(image_path).convert('RGBA')
    # chk_group을 통해 배경 제거와 스케일링 명령을 강제로 활성화합니다.
    # Rescale 옵션을 제거하여 투명도(Alpha) 채널을 보존합니다.
    processed_image_highres, _ = preprocess(predictor, input_image, chk_group=["Background Removal"])

    # 4. 6방향 도면 생성!
    print("[Headless] 6방향 2D 도면 및 노멀맵 생성 중... (RTX 4070 풀가동)")
    run_pipeline(
        pipeline=pipeline,
        cfg=cfg,
        single_image=processed_image_highres,
        guidance_scale=1.0, # UI의 기본값들
        steps=50,
        seed=42,
        crop_size=192,
        chk_group=["Write Results"] # 파일로 저장하라는 핵심 플래그
    )
    print("\n[SUCCESS] 도면 생성이 완료되었습니다! (outputs/ 폴더를 확인하세요)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("사용법: python headless_mv.py <이미지_경로>")