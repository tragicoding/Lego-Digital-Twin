# apps/frontend

MINIVERSE 전시용 태블릿 UI.
관객이 캐릭터와 오브제를 촬영하고 이름을 입력한 뒤 3D 변환 로딩을 확인하는 화면입니다.

---

## 화면 흐름

```
/start       → 로고 애니메이션 + 시작하기 버튼 (세션 생성)
/capture     → 안내 문구 → 캐릭터 촬영 → 안내 문구 → 오브제 촬영
/profile     → 캐릭터 이름 입력 → 오브제 이름 입력
/loading     → 3D 변환 진행 상태 (character + object 병렬)
             → ready_for_unity = true 시 완료 화면
```

---

## 개발 환경

- OS: Linux / WSL Ubuntu
- Node.js: v20+
- Framework: React 19, Vite, TypeScript, Tailwind CSS
- 주요 라이브러리: Framer Motion, Zustand, Axios, React Router

---

## 실행 방법

```bash
cd apps/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 3000
```

브라우저: `http://localhost:3000`

---

## API 서버 주소 설정

`apps/frontend/.env` 또는 환경 변수:

```
VITE_API_URL=http://localhost:8000
```

미설정 시 기본값 `http://localhost:8000` 사용.

---

## 디렉토리 구조

```
apps/frontend/src/
├── api/
│   └── client.ts           # Axios API 함수 (createSession, uploadAsset 등)
├── components/
│   ├── AppLayout.tsx        # 공통 레이아웃 (max-w-md 중앙 정렬)
│   └── PrimaryButton.tsx    # 공통 버튼 컴포넌트
├── pages/
│   ├── StartPage.tsx        # 로고 + 시작하기
│   ├── CapturePage.tsx      # 안내 문구 + 촬영
│   ├── ProfilePage.tsx      # 이름 입력
│   └── LoadingPage.tsx      # 진행 상태 + 완료
├── store/
│   └── sessionStore.ts      # Zustand 전역 상태
└── index.css
```

---

## Git 전략

```bash
git checkout feature/frontend
git fetch origin && git merge origin/develop
# ... 작업 ...
git add <파일명>
git commit -m "feat(frontend): ..."
git push origin feature/frontend
# GitHub에서 feature/frontend → develop PR 생성
```
