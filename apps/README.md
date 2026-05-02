# apps/

서버 사이드 애플리케이션 모음. Vision Engine, Backend, Hardware Controller로 구성됩니다.

```
apps/
├── engine-vision/   # 카메라 인식 및 3D 재구성
├── backend/         # Unity ↔ Vision Engine 통신 미들웨어
└── hardware/        # Arduino 하드웨어 제어
```

각 모듈의 상세 내용은 해당 디렉토리의 README를 참고하세요.

## 개발 환경 공통 사항

- OS: WSL2 (Ubuntu 24.04)
- Python: 3.9 (Conda 가상환경 권장)
- 환경 설정: `docs/environment.yml` 참고

```bash
conda env create -f docs/environment.yml
conda activate wonder3d
```
