#!/bin/bash
# TripoSR conda 환경 설정 스크립트
# RTX4070 Laptop (SM89), CUDA 11.8, WSL2, gcc-11 기준

set -e

ENV_NAME="triposr"
TRIPOSR_DIR="$(cd "$(dirname "$0")/TripoSR" && pwd)"

echo "=== TripoSR 환경 설정 시작 ==="
echo "TripoSR 경로: $TRIPOSR_DIR"

# 기존 환경 제거 (있으면)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "기존 ${ENV_NAME} 환경 제거 중..."
    conda env remove -n "$ENV_NAME" -y
fi

# conda 환경 생성 (Python 3.10)
echo "conda 환경 생성: $ENV_NAME (Python 3.10)"
conda create -n "$ENV_NAME" python=3.10 -y

# CUDA 11.8 + PyTorch 2.1.0 설치
echo "PyTorch 2.1.0 + CUDA 11.8 설치 중..."
conda run -n "$ENV_NAME" pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# NumPy <2.0 고정 (PyTorch 2.1.0 호환)
conda run -n "$ENV_NAME" pip install "numpy<2.0"

# conda env 내 gcc → gcc-11 심링크
# (CUDA 11.8은 gcc>11 미지원, cmake ID 컴파일 시 PATH에서 찾는 gcc를 강제 지정)
CONDA_BIN="/home/jskim/anaconda3/envs/${ENV_NAME}/bin"
ln -sf /usr/bin/gcc-11 "$CONDA_BIN/gcc"
ln -sf /usr/bin/g++-11 "$CONDA_BIN/g++"
ln -sf /usr/bin/gcc-11 "$CONDA_BIN/cc"

# torchmcubes 빌드용 도구 설치
conda run -n "$ENV_NAME" pip install "cmake==3.26.4" scikit-build-core pybind11 ninja

# torchmcubes 소스 클론 후 --no-build-isolation 빌드
# (pip install cmake==3.26.4 → 시스템 cmake 3.26 사용, Windows PATH 제거)
git clone https://github.com/tatsy/torchmcubes.git /tmp/torchmcubes_build || true
CLEAN_PATH="${CONDA_BIN}:/usr/local/cuda-11.8/bin:/usr/bin:/bin"

echo "torchmcubes 빌드 중 (SM89, gcc-11, cmake 3.26)..."
conda run -n "$ENV_NAME" \
    env CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
    TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 MAX_JOBS=4 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    PATH="$CLEAN_PATH" \
    CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-11" \
    pip install /tmp/torchmcubes_build --no-build-isolation

# 나머지 requirements 설치 (torchmcubes 제외)
echo "나머지 패키지 설치 중..."
conda run -n "$ENV_NAME" pip install \
    "omegaconf==2.3.0" \
    "Pillow==10.1.0" \
    "einops==0.7.0" \
    "transformers==4.35.0" \
    "trimesh==4.0.5" \
    "rembg" \
    "huggingface-hub" \
    "imageio[ffmpeg]" \
    "xatlas==0.0.9" \
    "moderngl==5.10.0" \
    "watchdog" \
    "gradio"

# rembg가 numpy 2.x로 업그레이드하는 경우 재고정
conda run -n "$ENV_NAME" pip install "numpy<2.0" --force-reinstall

echo ""
echo "=== 설치 완료 ==="
echo "환경 활성화: conda activate $ENV_NAME"
echo "파이프라인 실행: conda run -n $ENV_NAME python scripts/watch_and_run.py"
