import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./components/AppLayout";
import StartPage from "./pages/StartPage";
import CapturePage from "./pages/CapturePage";
import LoadingPage from "./pages/LoadingPage";

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Navigate to="/start" replace />} />
          <Route path="/start" element={<StartPage />} />
          <Route path="/capture" element={<CapturePage />} />
          <Route path="/loading" element={<LoadingPage />} />

          <Route path="/test" element={<Navigate to="/test/start" replace />} />
          <Route path="/test/start" element={<StartPage testMode />} />
          <Route path="/test/capture" element={<CapturePage testMode />} />
          <Route path="/test/loading" element={<LoadingPage testMode />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
}
