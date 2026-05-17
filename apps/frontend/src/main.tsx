/*
React 앱의 시작점(entry point)인 main.tsx 파일.
Vite + React 프로젝트에서는 main.tsx 또는 main.jsx
프론트엔드 앱을 브라우저 화면에 처음 띄우는 진입점
index.html의 root 영역에 React의 <App /> 컴포넌트를 연결해서
전체 웹앱을 실행시키는 시작 코드

index.html의 <div id="root"></div>
↓
main.tsx 실행
↓
createRoot가 root div를 찾음
↓
<App /> 컴포넌트를 렌더링
↓
React 앱 화면이 브라우저에 표시됨
*/


//React 개발 모드에서 문제를 더 빨리 찾기 위한 도구
import { StrictMode } from 'react'

//React 앱을 실제 HTML 화면에 붙이기 위한 함수
//React 컴포넌트는 그냥 코드만으로는 화면에 안 보이고, 
// HTML의 특정 위치에 렌더링해야한다.
import { createRoot } from 'react-dom/client'

//전체 앱에 적용할 CSS 파일을 가져온다.
import './index.css'
//App.tsx 파일에서 기본으로 export한 App 컴포넌트를 가져온다.
import App from './App.tsx'


//index.html 파일의 <div id="root"></div> 요소를 찾아서,
//그 위치에 React 앱을 렌더링한다.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
