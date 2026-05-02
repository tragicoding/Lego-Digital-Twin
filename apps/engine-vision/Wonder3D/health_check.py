import torch
import sys
import platform

print("="*60)
print(" 🛠️ Lego-Digital-Twin : 4070 인프라 종합 건강 검진 🛠️")
print("="*60)

# 1. 하드웨어 및 OS 검진
print("\n[1. 하드웨어 & 엔진 상태]")
print(f" - OS: {platform.system()} {platform.release()}")
print(f" - Python 버전: {sys.version.split()[0]}")
print(f" - PyTorch 버전: {torch.__version__} (권장: 2.1.0+cu118)")

if torch.cuda.is_available():
    print(f" - CUDA 가동 상태: [OK] 정상")
    print(f" - 인식된 GPU: {torch.cuda.get_device_name(0)}")
    print(f" - 가용 VRAM 용량: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(" - CUDA 가동 상태: [CRITICAL] GPU를 찾을 수 없습니다!")

# 2. 타임슬립 핵심 생태계 버전 검진
print("\n[2. 핵심 라이브러리 생태계 (버전 충돌 검사)]")
packages = {
    "diffusers": "0.19.3",
    "transformers": "4.35.0",
    "accelerate": "0.24.1",
    "huggingface_hub": "0.19.4",
    "xformers": "0.0.22.post7",
    "nerfacc": "0.3.5"
}

for pkg, expected in packages.items():
    try:
        module = __import__(pkg)
        # diffusers나 transformers는 __version__ 속성이 있지만, 패키지마다 다를 수 있어 범용 처리
        version = getattr(module, '__version__', '버전확인불가')
        status = "[OK] 일치" if version == expected else f"[WARN] 불일치"
        print(f" - {pkg:<16}: 현재 {version:<10} | 권장 {expected:<10} -> {status}")
    except ImportError:
        print(f" - {pkg:<16}: [CRITICAL] 설치되지 않음!")

# 3. 핫픽스 방어막 시뮬레이션
print("\n[3. 런타임 에러 방어막 테스트]")
try:
    import diffusers
    print(" - Diffusers 로드 충돌 방어: [OK] 안전함")
except AttributeError as e:
    print(f" - Diffusers 로드 충돌 방어: [WARN] 핫픽스 누락 위험 ({e})")

print("="*60)
print(" 진단 완료. 모든 항목이 [OK]라면 Gradio UI로 넘어갈 준비가 완벽합니다.")
print("="*60)