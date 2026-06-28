import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { createSession } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

interface StartPageProps {
  testMode?: boolean;
}

export default function StartPage({ testMode = false }: StartPageProps) {
  const navigate = useNavigate();
  const setSessionId = useSessionStore((s) => s.setSessionId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async () => {
    setLoading(true);
    setError(null);

    if (testMode) {
      setSessionId("test-session");
      navigate("/test/capture");
      return;
    }

    try {
      const res = await createSession();
      setSessionId(res.data.session_id);
      navigate("/capture");
    } catch {
      setError("서버 연결에 실패했습니다. 다시 시도해 주세요.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-white flex flex-col items-center justify-center px-6">
      <div className="relative w-full">
        <motion.img
          src="/images/miniverse-logo-crop.png"
          alt="MINIVERSE"
          className="w-full object-contain"
          initial={{ opacity: 0, scale: 0.85, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.65, ease: [0.34, 1.56, 0.64, 1] }}
        />

        <motion.img
          src="/images/block.png"
          alt=""
          className="absolute pointer-events-none"
          style={{
            width: "13%",
            right: "-9%",
            top: "-48%",
            mixBlendMode: "multiply",
          }}
          initial={{ y: -300, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{
            delay: 0.6,
            type: "spring",
            stiffness: 340,
            damping: 7,
            mass: 0.6,
          }}
        />
      </div>

      <motion.p
        className="text-gray-500 text-base text-center leading-relaxed mt-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.5 }}
      >
        나만의 레고가
        <br />
        디지털 세상에 등장합니다
      </motion.p>

      <motion.div
        className="w-3/4 mt-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.3, duration: 0.5 }}
      >
        <PrimaryButton onClick={handleStart} disabled={loading}>
          {loading ? "연결 중..." : "시작하기"}
        </PrimaryButton>
        {error && <p className="text-red-500 text-sm text-center mt-3">{error}</p>}
      </motion.div>
    </div>
  );
}
