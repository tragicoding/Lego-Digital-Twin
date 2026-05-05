# apps/

서버 사이드 애플리케이션 모음. Backend, Hardware Controller로 구성됩니다.

```
apps/
├── backend/tripo/   # 중앙 백엔드 서버 (카메라 → Tripo 3D → Unity)
└── hardware/        # Arduino 하드웨어 제어
```

각 모듈의 상세 내용은 해당 디렉토리의 README를 참고하세요.

## 개발 환경 공통 사항

- OS: WSL2 (Ubuntu 24.04)
- Python: 3.11 (Conda 가상환경: `triposr`)
- 백엔드 실행: `conda run -n triposr uvicorn apps.backend.tripo.main:app --host 0.0.0.0 --port 8000 --reload`
