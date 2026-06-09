# apps/

서버 사이드 및 클라이언트 애플리케이션 모음.
Backend (FastAPI), Frontend (React), Hardware Controller로 구성됩니다.

---

## 구조

```
apps/
├── backend/    # FastAPI 서버, PostgreSQL, Redis/RQ Worker, Tripo API
├── frontend/   # 태블릿 UI (React + Vite + TypeScript)
└── hardware/   # Arduino 제어 (예비)
```

---

## 개발 환경

- OS: Linux / WSL Ubuntu
- Python: 3.11 (conda 환경: `triposr`)
- Node.js: v20+

---

## 빠른 실행

```bash
# Docker (PostgreSQL + Redis)
docker compose up -d db redis

# Backend
conda run -n triposr uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000

# Worker
conda run -n triposr python run_worker.py

# Frontend
cd apps/frontend && npm run dev -- --host 0.0.0.0 --port 3000
```

각 모듈 상세는 하위 README를 참고하세요.
- [backend/README.md](backend/README.md)
- [frontend/README.md](frontend/README.md)

---

## Git 전략

- `feature/backend` 브랜치에서 backend 작업
- `feature/frontend` 브랜치에서 frontend 작업
- 실험 또는 기획 검증은 `test_app` 브랜치 사용
- 작업 완료 후 `develop`으로 PR 생성
