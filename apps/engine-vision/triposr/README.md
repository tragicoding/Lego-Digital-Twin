# TripoSR — Windows 카메라 → WSL2 자동 3D 재구성

RTX4070 Laptop (SM89), CUDA 11.8, WSL2 환경 기준.

## 환경 요구사항

| 항목 | 버전 |
|------|------|
| GPU | RTX4070 (SM89) / 8GB VRAM |
| CUDA | 11.8 |
| GCC | 11 (torchmcubes 빌드용) |
| Python | 3.10 |
| PyTorch | 2.1.0 + cu118 |

## 설치

```bash
# 1. conda 환경 생성 및 의존성 설치 (약 10~15분)
cd apps/engine-vision/triposr
chmod +x setup_env.sh
./setup_env.sh

# 2. 환경 활성화
conda activate triposr
```

## 사용법

### 자동 파이프라인 (Windows 카메라 → 자동 3D 변환)

```bash
conda activate triposr
python scripts/watch_and_run.py
```

Windows Camera 앱으로 사진 촬영 시 자동으로 감지하여 3D 메시 생성.
- 감시 폴더: `C:\Users\wlstj\OneDrive\사진\Camera Roll\`
- 출력: `triposr/output/<이미지명_타임스탬프>/`

### 수동 단일 이미지 처리

```bash
conda activate triposr
python scripts/run_single.py <이미지경로>

# 예시
python scripts/run_single.py input/my_object.jpg
python scripts/run_single.py input/lego.jpg --mc-resolution 384 --remove-bg
```

## 파라미터 튜닝 (8GB VRAM 기준)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--mc-resolution` | 256 | Marching Cubes 해상도. 384로 올리면 더 세밀 (VRAM 주의) |
| `--chunk-size` | 8192 | 청크 크기. OOM 발생 시 4096으로 낮춤 |
| `--remove-bg` | OFF | 배경 제거. 단색 배경 촬영 시 권장 |

## 디렉토리 구조

```
triposr/
├── TripoSR/          # TripoSR 원본 소스
├── input/            # 처리할 입력 이미지
├── output/           # 생성된 3D 메시 (.obj, .glb)
├── logs/             # 파이프라인 로그
├── scripts/
│   ├── watch_and_run.py  # 자동 파이프라인
│   └── run_single.py     # 수동 단발성 실행
└── setup_env.sh      # 환경 설치 스크립트
```

## 출력 파일

TripoSR는 이미지당 다음 파일을 생성합니다:
- `mesh.obj` — 3D 메시 (Wavefront OBJ)
- `mesh.glb` — 3D 메시 (GLTF Binary, 뷰어 호환)
