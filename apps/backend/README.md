# apps/backend

MINIVERSE 전시 시스템의 백엔드 서버.
관객 세션 관리, 이미지 업로드, Tripo3D API 연동, Unity 데이터 제공을 담당합니다.

---

## 역할

```
[Frontend]
  이미지 업로드 (character / object)
        │
        ▼
[FastAPI]
  - DB 세션 생성 / 프로필 저장
  - 이미지 저장 → Redis Queue 등록
        │
        ▼
[RQ Worker × 2 (병렬)]
  character: image_to_model → rig → retarget (walk + idle 동시)
  object:    image_to_model → GLB 저장
        │
        ▼
[WebSocket]
  session_ready 이벤트 → Unity 전송
        │
        ▼
[Unity]
  GET /unity/sessions/{id} → GLB URL 수신 → 모델 로드
```

---

## 개발 환경

- OS: Linux / WSL Ubuntu
- Python: 3.11 (conda 환경: `triposr`)
- Framework: FastAPI 0.111.0
- DB: PostgreSQL 16 (Docker)
- Cache / Queue: Redis 7 (Docker)
- Worker: RQ 2.8.0
- 3D API: Tripo3D API

---

## 디렉토리 구조

```
apps/backend/
├── api/
│   └── routers/        # sessions, assets, unity_data, ws_router
├── core/               # config, database
├── models/             # SQLAlchemy ORM (session, asset, asset_animation)
├── schemas/            # Pydantic (session, asset, unity)
├── services/           # tripo_service, queue_service, ws_manager, event_service
├── worker/
│   └── tasks/          # character_task.py, object_task.py
├── storage/
│   ├── images/         # 업로드된 이미지 (git 제외)
│   └── models/         # 생성된 GLB 파일 (git 제외)
├── .env.example
├── main.py
└── requirements.txt
```

---

## 실행 방법

### 1. 의존성 설치

```bash
conda activate triposr
pip install -r apps/backend/requirements.txt
```

### 2. 환경 변수 설정

```bash
cp apps/backend/.env.example apps/backend/.env
```

`.env` 필수 항목:
```
TRIPO_API_KEY=your_tripo_api_key
DATABASE_URL=postgresql://lego:legopass@localhost:5432/lego_twin
REDIS_URL=redis://localhost:6379
STORAGE_BASE=apps/backend/storage
BACKEND_HOST=http://localhost:8000
```

### 3. Docker (PostgreSQL + Redis) 실행

```bash
docker compose up -d db redis
```

### 4. DB 마이그레이션

```bash
conda run -n triposr alembic upgrade head
```

### 5. FastAPI 서버 실행

```bash
conda run -n triposr uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000
```

### 6. RQ Worker 실행 (프로젝트 루트에서)

```bash
conda run -n triposr python run_worker.py
```

Worker 2개가 병렬 실행됩니다:
- `lego-character`: 3D 생성 + 리깅 + 애니메이션
- `lego-object`: 3D 생성만

---

## 주요 API

| Method | URL | 설명 |
|---|---|---|
| POST | `/sessions` | DB 세션 생성 |
| PATCH | `/sessions/{id}/profile` | 프로필 / 말풍선 저장 |
| PATCH | `/sessions/{id}/names` | Unity 대기화면에서 입력한 캐릭터/오브제 이름 저장 |
| POST | `/sessions/{id}/assets` | 이미지 업로드 (character / object, front/left/back/right) |
| GET | `/sessions/{id}/status` | 처리 상태 polling |
| GET | `/unity/sessions/{id}` | Unity용 완성 데이터 |
| WS | `/ws/unity` | session_ready 이벤트 수신 |

---

## Git 전략

```bash
git checkout feature/backend
git fetch origin && git merge origin/develop
# ... 작업 ...
git add -A
git commit -m "feat(backend): ..."
git push origin feature/backend
# GitHub에서 feature/backend → develop PR 생성
```
