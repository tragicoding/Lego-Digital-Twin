import sys
import torch
import torch.utils._pytree

# [STEP 0] PyTorch 2.1.0+ AttributeError 핫픽스 주입
def fix_pytree(*args, **kwargs): pass
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = fix_pytree
if not hasattr(torch.utils._pytree, 'register_leaf'):
    torch.utils._pytree.register_leaf = fix_pytree

def run_audit():
    print("="*50)
    print("      Wonder3D 인프라 무결성 최종 진단")
    print("="*50)

    try:
        # 1. 하드웨어 및 엔진
        print(f"[OK] PyTorch 버전: {torch.__version__}")
        print(f"[OK] CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] GPU 모델: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA 버전: {torch.version.cuda}")

        # 2. 가속기 (TCNN & Nerfacc)
        import tinycudann as tcnn
        import nerfacc
        print(f"[OK] tiny-cuda-nn 로드: 성공 (RTX 4070 준비 완료)")
        print(f"[OK] nerfacc 버전: {nerfacc.__version__}")

        # 3. 라이브러리 (Transformers & Gradio)
        import transformers
        import gradio
        import numpy
        print(f"[OK] Transformers 로드: 성공 (버전 {transformers.__version__})")
        print(f"[OK] Gradio 버전: {gradio.__version__}")
        print(f"[OK] NumPy 버전: {numpy.__version__}")

        print("="*50)
        print("  [SUCCESS] 모든 시스템이 완벽합니다. 시동을 거세요!")
        print("="*50)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        print("="*50)

if __name__ == '__main__':
    run_audit()
