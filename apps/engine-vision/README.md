# engine-vision

레고 모형을 카메라로 촬영하여 3D 메시(.obj)로 재구성하는 Vision Engine입니다.
생성된 3D 모델은 Backend를 통해 Unity 클라이언트로 전달됩니다.

---

## 파이프라인

```
카메라 촬영 (입력 이미지)
      │
      ▼
SAM (Segment Anything Model)
  └─ 레고 모형 배경 제거 / 마스킹
      │
      ▼
Wonder3D (Multi-view Diffusion)
  └─ 단일 이미지 → 6방향 멀티뷰 이미지 + 법선 맵 생성
      │
      ▼
instant-nsr-pl (Neural Surface Reconstruction)
  └─ 멀티뷰 → 3D 메시(.obj) 재구성
      │
      ▼
.obj 파일 → Backend 전송
```

---

## 기술 스택

| 기술 | 버전 | 용도 |
|---|---|---|
| Python | 3.9 | 런타임 |
| PyTorch | 2.1.0+cu118 | 딥러닝 프레임워크 |
| CUDA | 11.8 | GPU 가속 |
| GCC/G++ | 11 | CUDA 호환 컴파일러 |
| SAM | ViT-H | 이미지 세그멘테이션 |
| Wonder3D | - | 멀티뷰 생성 |
| instant-nsr-pl | - | 3D 재구성 |
| tiny-cuda-nn | - | NeRF 가속 (수동 빌드) |
| nerfacc | 0.3.5 | NeRF 가속 보조 |

---

## 환경 설정

### 1. Conda 환경 생성

```bash
conda env create -f ../../docs/environment.yml
conda activate wonder3d
```

### 2. PyTorch (CUDA 11.8) 설치

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. GCC 버전 고정 (CUDA 11.8 호환)

```bash
sudo update-alternatives --set gcc /usr/bin/gcc-11
sudo update-alternatives --set g++ /usr/bin/g++-11
```

### 4. tiny-cuda-nn 수동 빌드

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

> **주의:** `torch.utils._pytree` 버전 불일치 에러 발생 시 `fix_check.py` 실행

---

## 실행 방법

### 입력 이미지 준비

```bash
# 카메라 캡처 이미지를 inputs/ 폴더에 저장
Wonder3D/inputs/lego_capture_YYYYMMDD_HHMMSS.png
```

### Wonder3D 실행 (멀티뷰 생성)

```bash
cd Wonder3D
python run_mvdiffusion_img.py \
    --config configs/mvdiffusion-joint-ortho-6views.yaml \
    validation_dataset.root_dir=./inputs \
    validation_dataset.filepaths=['your_image.png'] \
    save_dir=./outputs
```

### instant-nsr-pl 실행 (3D 재구성)

```bash
cd Wonder3D/instant-nsr-pl
python launch.py \
    --config configs/neuralangelo-ortho-wmask.yaml \
    --gpu 0 \
    --train \
    dataset.root_dir=../outputs/cropsize-192-cfg1.0/ \
    dataset.scene=<씬이름> \
    dataset.num_workers=0
```

### 결과물

```
instant-nsr-pl/exp/<scene>/<run>/save/
├── it0-mc192.obj       # 초기 메시
└── it3000-mc192.obj    # 최종 재구성 메시
```

---

## 개발 브랜치

`feature/vision` 브랜치에서 작업 후 `develop`으로 PR

```bash
git checkout feature/vision
git fetch origin && git merge origin/develop   # 최신 동기화
# ... 작업 ...
git push origin feature/vision
# GitHub에서 feature/vision → develop PR 생성
```

---

## 알려진 이슈

| 이슈 | 해결책 |
|---|---|
| `torch.utils._pytree` AttributeError | `fix_check.py` 실행 |
| single GPU에서 DDP 데드락 | `launch.py` strategy=None 패치 적용됨 |
| DataLoader workers 멈춤 | `dataset.num_workers=0` 옵션 추가 |
| `pin_memory` CUDA 에러 | `datasets/ortho.py` pin_memory=False 패치 적용됨 |



