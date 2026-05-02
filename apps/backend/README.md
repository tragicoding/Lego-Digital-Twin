# backend

Vision Engine, Unity 클라이언트, Hardware Controller 사이의 데이터를 중계하는 미들웨어 서버입니다.

---

## 역할

```
engine-vision (.obj 생성)
      │
      ▼
   backend  ←→  hardware (Arduino: 조명, 박수 감지)
      │
      ▼
unity-client (3D 모델 수신, 대기 변경, 이벤트 수신)
```

| 기능 | 설명 |
|---|---|
| 3D 모델 전달 | Vision Engine이 생성한 `.obj`를 Unity로 전송 |
| 대기 동기화 | 물리 세계 조명 값 → Unity atmosphere 파라미터 변환 |
| 이벤트 중계 | 박수 감지 신호 → Unity 불꽃 이벤트 트리거 |

---

## 개발 예정 스택

> 현재 설계 단계. 확정 후 업데이트 예정.

- **언어:** (미정)
- **통신:** WebSocket / REST API
- **Unity 연동:** Unity Netcode 또는 커스텀 소켓

---

## 개발 브랜치

`feature/backend` 브랜치에서 작업 후 `develop`으로 PR

```bash
git checkout feature/backend
git fetch origin && git merge origin/develop
# ... 작업 ...
git push origin feature/backend
# GitHub에서 feature/backend → develop PR 생성
```

---

## API 설계 (초안)

| Endpoint / Event | 방향 | 데이터 | 설명 |
|---|---|---|---|
| `POST /model/upload` | Vision → Backend | `.obj` 파일 | 3D 모델 업로드 |
| `WS /unity/model` | Backend → Unity | 모델 바이너리 | Unity에 모델 전달 |
| `WS /unity/atmosphere` | Backend → Unity | `{type: "night"}` | 대기 변경 |
| `WS /unity/fireworks` | Backend → Unity | `{trigger: true}` | 불꽃 이벤트 |
| `POST /hardware/event` | Hardware → Backend | `{type: "clap" \| "light", value}` | 하드웨어 이벤트 수신 |
