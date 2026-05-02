# ====================================================================
# [STEP 0] RTX 4070 & PyTorch 2.1.0 호환성 통합 핫픽스 (방어막)
# ====================================================================
import torch
import torch.utils._pytree
import torch.distributed

# 1. _pytree 에러 방어
def fix_pytree(*args, **kwargs): pass
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = fix_pytree
if not hasattr(torch.utils._pytree, 'register_leaf'):
    torch.utils._pytree.register_leaf = fix_pytree

# 2. xpu 에러 방어
if not hasattr(torch, "xpu"):
    class DummyXPU:
        def __init__(self):
            self.device_count = lambda: 0
            self.is_available = lambda: False
        def empty_cache(self): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = DummyXPU()

# 3. device_mesh 에러 방어
if not hasattr(torch.distributed, "device_mesh"):
    class DummyMesh:
        DeviceMesh = None
    torch.distributed.device_mesh = DummyMesh()

# ====================================================================
# [STEP 1] Wonder3D 실제 구동 코드 (엔진)
# ====================================================================
import requests
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import DiffusionPipeline

print("="*50)
print("  RTX 4070 Wonder3D 단독 엔진 시동 중...")
print("="*50)

def load_wonder3d_pipeline():
    print("[INFO] 허깅페이스 서버에서 모델을 확인하고 로드합니다...")
    # trust_remote_code=True 를 추가하여 커스텀 파이프라인 에러를 방지합니다.
    pipeline = DiffusionPipeline.from_pretrained(
        'flamehaze1115/wonder3d-v1.0', 
        custom_pipeline='flamehaze1115/wonder3d-pipeline',
        torch_dtype=torch.float16,
        trust_remote_code=True 
    )

    # 4070 VRAM 최적화 활성화
    pipeline.unet.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
        print(f"[OK] GPU 로드 성공: {torch.cuda.get_device_name(0)}")
    return pipeline

try:
    # 1. 엔진 로드 (이때 가중치가 없으면 자동으로 다운로드 됩니다)
    pipeline = load_wonder3d_pipeline()

    # 2. 샘플 이미지 준비
    print("[INFO] 테스트용 샘플 이미지를 가져옵니다...")
    img_url = "https://d.skis.ltd/nrp/sample-data/lysol.png"
    cond = Image.open(requests.get(img_url, stream=True).raw)
    cond = Image.fromarray(np.array(cond)[:, :, :3])

    # 3. 3D 시점 생성 (추론)
    print("[WAIT] 4070 텐서 코어 가동 중... (약 10~20초 소요)")
    images = pipeline(cond, num_inference_steps=20, output_type='pt', guidance_scale=1.0).images

    # 4. 결과물 저장
    result = make_grid(images, nrow=6, padding=0, value_range=(0, 1))
    save_image(result, 'test_result.png')

    print("="*50)
    print("[SUCCESS] 완벽합니다! 결과물이 'test_result.png'로 저장되었습니다.")
    print("="*50)

except Exception as e:
    print(f"\n[CRITICAL ERROR] {e}")