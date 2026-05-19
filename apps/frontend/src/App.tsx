import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AppLayout from "./components/AppLayout";
import StartPage from "./pages/StartPage";
import CapturePage from "./pages/CapturePage";
import ProfilePage from "./pages/ProfilePage";
import LoadingPage from "./pages/LoadingPage";

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Navigate to="/start" replace />} />
          <Route path="/start" element={<StartPage />} />
          <Route path="/capture" element={<CapturePage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/loading" element={<LoadingPage />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
}
