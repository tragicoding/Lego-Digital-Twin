import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import StartPage from "./pages/StartPage";
import CapturePage from "./pages/CapturePage";
import ProfilePage from "./pages/ProfilePage";
import LoadingPage from "./pages/LoadingPage";
import TestPage from "./pages/TestPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/start" replace />} />
        <Route path="/start" element={<StartPage />} />
        <Route path="/capture" element={<CapturePage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="/loading" element={<LoadingPage />} />
        <Route path="/test" element={<TestPage />} />
      </Routes>
    </BrowserRouter>
  );
}
