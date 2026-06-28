# apps/frontend

MINIVERSE tablet/kiosk UI built with React, Vite, TypeScript, Tailwind CSS, Framer Motion, Zustand, and Axios.

## Routes

```text
/start    create a visitor session
/capture  capture/submit character and object steps
/loading  move the completed visitor session into the Unity queue
/test/start    review/demo mode without backend calls
/test/capture  fake capture flow without DB or workers
/test/loading  fake Unity move screen
```

## Development

```bash
cd apps/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 3000
```

Open `http://localhost:3000`.

## API URL

Set the backend URL in `apps/frontend/.env.local`:

```text
VITE_API_URL=http://localhost:8000
```

When the app is opened from `localhost` or `127.0.0.1`, it uses `http://localhost:8000`.

## Source Layout

```text
apps/frontend/src/
  api/client.ts
  components/AppLayout.tsx
  components/PrimaryButton.tsx
  pages/StartPage.tsx
  pages/CapturePage.tsx
  pages/LoadingPage.tsx
  store/sessionStore.ts
  index.css
```
