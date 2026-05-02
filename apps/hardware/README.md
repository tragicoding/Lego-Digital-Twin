# hardware

Arduino 기반 하드웨어 제어 모듈입니다.
물리 세계의 조명 조절과 박수 감지를 담당하며, 이를 Backend 서버로 전송합니다.

---

## 담당 기능

| 기능 | 센서/액추에이터 | 가상 세계 반응 |
|---|---|---|
| 조명 밝기 조절 | 조도 센서 / 가변저항 | Unity atmosphere 변경 (낮/노을/밤/흐림/눈/비) |
| 박수 감지 | 사운드 센서 | Unity 불꽃 축제 트리거 |

---

## 기술 스택

- **MCU:** Arduino (Uno / Nano)
- **언어:** C++ (Arduino IDE)
- **통신:** Serial → Backend 서버 (USB / Bluetooth 예정)

---

## 하드웨어 구성 (예정)

```
Arduino
├── A0  ← 조도 센서 (아날로그)
├── D2  ← 사운드 센서 (디지털)
└── TX  → Serial → Backend 서버
```

---

## 통신 프로토콜 (초안)

Arduino → Backend로 JSON 직렬 전송

```json
{ "type": "light", "value": 75 }
{ "type": "clap" }
```

`value` 범위 (0~100):

| 값 | Unity 대기 |
|---|---|
| 90~100 | 맑은 낮 |
| 70~89 | 노을 |
| 50~69 | 흐림 |
| 20~49 | 밤 |
| 특수 트리거 | 눈 / 비 |

---

## 개발 브랜치

`feature/hardware` 브랜치에서 작업 후 `develop`으로 PR

```bash
git checkout feature/hardware
git fetch origin && git merge origin/develop
# ... 작업 ...
git push origin feature/hardware
# GitHub에서 feature/hardware → develop PR 생성
```

---

## 빌드 및 업로드

```bash
# Arduino IDE 사용 시
# 1. hardware/ 폴더의 .ino 파일 열기
# 2. 보드: Arduino Uno 선택
# 3. 포트 선택 후 업로드 (Ctrl+U)
```

> `.hex`, `.elf` 빌드 결과물은 `.gitignore`에 의해 추적되지 않습니다.
