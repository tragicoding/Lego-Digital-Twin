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
from PIL import Image, ImageFilter, ImageEnhance
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
    processed_image_highres, _ = preprocess(predictor, input_image, chk_group=["Background Removal"])

    # 3-1. 레고 디테일 강화 (스터드·모서리 윤곽 선명화)
    print("[Headless] 레고 엣지 샤프닝 적용 중...")
    r, g, b, a = processed_image_highres.split()
    rgb = Image.merge("RGB", (r, g, b))
    # 언샤프 마스크: 엣지를 뚜렷하게
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))
    # 대비 강화: 스터드 음영 부각
    rgb = ImageEnhance.Contrast(rgb).enhance(1.4)
    # 선명도 추가 강화
    rgb = ImageEnhance.Sharpness(rgb).enhance(2.5)
    r2, g2, b2 = rgb.split()
    processed_image_highres = Image.merge("RGBA", (r2, g2, b2, a))

    # 4. 6방향 도면 생성!
    print("[Headless] 6방향 2D 도면 및 노멀맵 생성 중... (RTX 4070 풀가동)")
    run_pipeline(
        pipeline=pipeline,
        cfg=cfg,
        single_image=processed_image_highres,
        guidance_scale=3.0,  # 1.0→3.0: 입력 이미지 디테일을 더 충실하게 반영
        steps=75,            # 50→75: 더 정교한 생성
        seed=42,
        crop_size=192,
        chk_group=["Write Results"]
    )
    print("\n[SUCCESS] 도면 생성이 완료되었습니다! (outputs/ 폴더를 확인하세요)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("사용법: python headless_mv.py <이미지_경로>")