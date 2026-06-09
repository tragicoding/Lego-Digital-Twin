# MINIVERSE — 레고 디지털 트윈
### 내가 만든 레고가 가상 세계의 주인공이 됩니다
### Your Lego creation becomes the star of a virtual world

---

## 무엇을 하나요? / What is it?
🇰🇷 관람객이 레고로 캐릭터와 오브제를 만들고 **태블릿으로 사진을 찍으면**, AI가 그 사진을
**3D 모델로 변환**해 화면 속 가상 세계 **MINIVERSE**에 등장시킵니다. 내가 만든 캐릭터가 직접
안내자가 되어 환영하고, 원하는 **동작을 입력**하면 그대로 움직이며, 다른 관람객의 작품에
**하트(투표)** 를 눌러 가장 인기 있는 작품을 함께 골라갑니다. *(PC 화면 기반 체험 — 별도 VR 장비 불필요)*

🇬🇧 Build a character and an object out of Lego, **snap a photo on a tablet**, and an AI turns it into a
**3D model** inside the on-screen virtual world **MINIVERSE**. Your own creation greets you as a guide,
performs any **motion you type**, and you **vote (heart)** on others' works to crown the most-loved one.
*(A screen-based PC experience — no VR headset required.)*

---

## 체험 흐름 / Experience flow
```
레고 제작  →  태블릿 촬영  →  AI 3D 변환  →  서버 등록  →  화면 속 세계에 등장  →  체험
Build         Photograph       AI 3D-gen      Server          Appears on screen        Play
```
1. **환영** — 내 캐릭터(가이드)가 다가와 인사하고 광장으로 순간이동 *(가이드 모드)*
2. **동작 체험** — "춤춰줘" 같은 명령 입력 → 캐릭터가 즉시 동작, 마음에 들면 **시그니처 동작**으로 저장
3. **투표** — 다른 작품에 다가가 하트 ❤️, 1위 작품엔 빛나는 ⭐ 별 *(자유 모드)*

조작: **마우스 우클릭+드래그**(시선 회전) · **WASD**(이동) · **Space**(점프) · **키보드**(동작 명령 입력)
Controls: right-drag to look · WASD to move · Space to jump · keyboard to type motion commands

---

## 기술 구조 / How it's built
🇰🇷 사진 한 장이 살아 움직이는 3D 캐릭터가 되기까지, 네 단계의 시스템이 맞물려 동작합니다.

**① 입력 — 태블릿 앱 (React)**
관람객이 작품을 촬영하고 이름을 입력하면 하나의 **세션(Session)** 데이터가 만들어집니다.

**② 처리 — 서버 (FastAPI · PostgreSQL · Redis)**
세션을 데이터베이스에 저장하고, **Redis 작업 큐**에 변환 작업을 등록해 백그라운드에서 병렬 처리합니다.
**Tripo3D AI** 가 2D 사진을 **3D 모델로 변환**하고, 캐릭터에는 움직임을 위한 **자동 리깅(뼈대 삽입)** 까지 수행합니다.

**③ 출력 — Unity 가상 세계 (Unity · C#)**
변환이 끝나면 Unity가 **REST API**로 세션 데이터를 받아 3D 모델을 **게임 실행 중 실시간으로 불러옵니다**
(캐릭터 FBX는 **TriLib**, 오브제 GLB는 **glTFast**). 좋아요는 **WebSocket** 으로 모든 화면에 실시간 반영됩니다.

**④ 애니메이션 — Mixamo + Unity Humanoid**
AI가 만든 캐릭터의 비표준 뼈대를 Unity 표준(22본)에 **자동 매핑**해, **Mixamo 모션**(춤·점프·손흔들기 등)을
어떤 캐릭터에도 적용합니다. 관람객이 입력한 단어를 동작으로 해석해(`MotionPromptParser`) 즉시 재생합니다.

🇬🇧 Four subsystems turn one photo into a living 3D character:
**(1) Tablet app (React)** creates a *Session* · **(2) Server (FastAPI · PostgreSQL · Redis)** queues the
job and calls **Tripo3D AI** for image-to-3D + **auto-rigging** · **(3) Unity** loads the model **at runtime**
(**TriLib** for FBX characters, **glTFast** for GLB objects) and syncs votes live over **WebSocket** ·
**(4)** AI bones are auto-mapped to Unity's 22-bone **Humanoid** rig so any **Mixamo** motion plays on demand.

---

## 사용 기술 / Tech at a glance
| 영역 / Area | 기술 / Technology |
|---|---|
| 가상 세계 / Virtual world | **Unity** · C# · URP |
| AI 3D 생성 / AI 3D-gen | **Tripo3D** (사진 → 3D + 자동 리깅) |
| 서버 / Backend | **FastAPI** · PostgreSQL · Redis · RQ Worker |
| 태블릿 앱 / Tablet app | **React** · TypeScript |
| 실시간 통신 / Realtime | **WebSocket** · REST API |
| 런타임 3D 로딩 / Runtime loading | **TriLib** (FBX) · **glTFast** (GLB) |
| 애니메이션 / Animation | **Mixamo** 모션 · Unity Humanoid 리타게팅 |

---

## 핵심 포인트 / Highlights
🇰🇷 **세상에 하나뿐인 작품** — 관람객의 실제 레고가 그대로 3D 캐릭터가 됩니다.
**살아 움직이는 캐릭터** — AI 자동 리깅 + Mixamo 모션으로 입력한 동작을 즉시 재생합니다.
**기술적 도전** — 게임 실행 중 3D 모델을 불러오는 **런타임 로딩**, AI가 만든 비표준 뼈대를 표준 골격에
맞추는 **본 매핑**, 모든 화면이 함께 갱신되는 **실시간 투표**가 이 프로젝트의 핵심 엔지니어링입니다.
**함께 만드는 광장** — 모든 관람객의 작품이 한 공간에 모여 실시간 투표로 인기 작품이 정해집니다.

🇬🇧 **One-of-a-kind** — a real Lego becomes a 3D character.
**It comes alive** — AI auto-rigging + Mixamo motions play your commands instantly.
**Engineering challenges** — **runtime model loading**, **bone re-mapping** of AI rigs onto a standard
skeleton, and **live multi-screen voting** are the core technical achievements.
**A shared plaza** — every visitor's work gathers in one space, ranked by live voting.

---

*MINIVERSE · Lego Digital Twin Project · 화면 기반(비VR) 빌드*
