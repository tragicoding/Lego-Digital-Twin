# Lego Digital Twin

> 물리 세계에서 조립한 레고 모형을 카메라로 인식하고, VR 레고 놀이공원 안에 그대로 재현하는 디지털 트윈 시스템

---

## 프로젝트 개요

사용자가 물리 보드 위에서 레고를 조립하면, 카메라가 이를 3D로 인식하여 Unity VR 환경에 실시간으로 재현합니다.
VR 헤드셋을 착용한 사용자는 레고 놀이공원 안에서 자신이 직접 만든 모형이 살아 숨쉬는 것을 경험하게 됩니다.

```
[물리 세계]                        [가상 세계 (VR)]
 사용자 레고 조립
      │
      ▼
 카메라 촬영 → Vision Engine → 3D 모델 생성
                                    │
                                    ▼
                             Unity 놀이공원에 배치
                                    │
                                    ▼
                          VR 헤드셋으로 경험
```

---

## 주요 인터랙션

| 물리 세계 입력 | 가상 세계 반응 |
|---|---|
| 레고 모형 조립 | Unity 공원 내 동일 모형 생성 |
| 조명 밝기 조절 | 대기 변화 (낮 / 노을 / 밤 / 흐림 / 눈 / 비) |
| VR 착용 중 박수 | 불꽃 축제 시작 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                     Lego Digital Twin                   │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ engine-vision│───▶│   backend    │───▶│unity-client│ │
│  │  (Python)    │    │  (Server)    │    │  (Unity)  │  │
│  └──────────────┘    └──────┬───────┘    └───────────┘  │
│                             │                           │
│                      ┌──────▼───────┐                   │
│                      │  hardware    │                   │
│                      │  (Arduino)   │                   │
│                      └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

| 모듈 | 역할 | 기술 스택 |
|---|---|---|
| `engine-vision` | 카메라 영상 처리, 3D 재구성 | Python, Wonder3D, PyTorch, SAM |
| `backend` | 모듈 간 통신, 데이터 중계 | (미정) |
| `unity-client` | VR 놀이공원 렌더링, 인터랙션 | Unity, C#, XR |
| `hardware` | 조명·박수 입력 감지 | Arduino, C++ |

---

## 개발 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop (Ada Lovelace) |
| OS | Windows + WSL2 (Ubuntu 24.04) |
| Python | 3.9 (Conda) |
| CUDA | 11.8 |
| Unity | 6000.x |

---

## 브랜치 전략

```
main                  ← 최종 배포 (직접 커밋 금지)
└── develop           ← 통합 브랜치 (모든 PR의 대상)
    ├── feature/unity     ← Unity 클라이언트
    ├── feature/vision    ← Vision Engine
    ├── feature/backend   ← 백엔드 서버
    ├── feature/hardware  ← 하드웨어 제어
    └── feature/test      ← 통합 테스트
```

### 기본 워크플로우

```bash
# 1. 작업 시작 전: develop 최신 내용 반영
git checkout feature/본인브랜치
git fetch origin
git merge origin/develop

# 2. 작업 후 push
git add .
git commit -m "feat(파트): 작업 내용"
git push origin feature/본인브랜치

# 3. GitHub에서 feature/* → develop PR 생성
```

### 커밋 메시지 규칙

```
feat(파트):     새 기능 추가
fix(파트):      버그 수정
refactor(파트): 리팩토링
docs(파트):     문서 수정
chore(파트):    빌드, 설정 변경
```

---

## 디렉토리 구조

```
Lego-Digital-Twin/
├── apps/
│   ├── engine-vision/   # Vision Engine (3D 재구성)
│   ├── backend/         # 미들웨어 서버
│   └── hardware/        # Arduino 제어
├── unity-client/        # Unity VR 클라이언트
├── docs/                # 환경 설정 문서
└── .gitignore
```

---

## 팀 구성

| 역할 | 담당 브랜치 |
|---|---|
| Vision Engine | `feature/vision` |
| Unity 클라이언트 | `feature/unity` |
| 백엔드 | `feature/backend` |
| 하드웨어 | `feature/hardware` |
