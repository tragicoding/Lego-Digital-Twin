import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { finalizeSession } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

export default function LoadingPage() {
  const { sessionId, reset } = useSessionStore();
  const navigate = useNavigate();
  const [showButton, setShowButton] = useState(false);
  const [moving, setMoving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setShowButton(true), 1200);
    return () => clearTimeout(timer);
  }, []);

  const handleMove = async () => {
    if (!sessionId || moving) return;
    setMoving(true);
    setError(null);
    try {
      await finalizeSession(sessionId);
      reset();
      navigate("/start");
    } catch {
      setError("세션 이동에 실패했습니다. 관리자에게 문의해주세요.");
      setMoving(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-8 bg-white text-center">
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        <p className="text-3xl font-black leading-snug text-gray-900 whitespace-pre-line">
          {"이제 MINIVERSE로 이동할게요!\n옆의 부스로 이동하세요!"}
        </p>

        {showButton && (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <PrimaryButton onClick={handleMove} disabled={moving}>
              {moving ? "이동 중..." : "이동하기"}
            </PrimaryButton>
          </motion.div>
        )}

        {error && <p className="text-sm text-red-500">{error}</p>}
      </motion.div>
    </div>
  );
}
