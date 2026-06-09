import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { createSession } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

export default function StartPage() {
  const navigate = useNavigate();
  const setSessionId = useSessionStore((s) => s.setSessionId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await createSession();
      setSessionId(res.data.session_id);
      navigate("/capture");
    } catch {
      setError("서버 연결에 실패했습니다. 다시 시도해주세요.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-white flex flex-col items-center justify-center px-6">

      {/* 로고 영역 — 텍스트 + 블록을 relative 컨테이너 안에서 겹쳐 배치 */}
      <div className="relative w-full">
        {/* MINIVERSE 텍스트 */}
        <motion.img
          src="/images/miniverse-logo-crop.png"
          alt="MINIVERSE"
          className="w-full object-contain"
          initial={{ opacity: 0, scale: 0.85, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.65, ease: [0.34, 1.56, 0.64, 1] }}
        />

        {/* 블록 — 위에서 공처럼 튀며 reference 위치로 낙하 */}
        <motion.img
          src="/images/block.png"
          alt=""
          className="absolute pointer-events-none"
          style={{
            width: "13%",
            right: "-9%",
            top: "-48%",
            mixBlendMode: "multiply",   // 흰 배경 투명 처리
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

      {/* 설명 문구 */}
      <motion.p
        className="text-gray-500 text-base text-center leading-relaxed mt-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.5 }}
      >
        나만의 레고가<br />디지털 세상에 등장합니다
      </motion.p>

      {/* 시작 버튼 */}
      <motion.div
        className="w-3/4 mt-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.3, duration: 0.5 }}
      >
        <PrimaryButton onClick={handleStart} disabled={loading}>
          {loading ? "연결 중..." : "시작하기"}
        </PrimaryButton>
        {error && (
          <p className="text-red-500 text-sm text-center mt-3">{error}</p>
        )}
      </motion.div>

    </div>
  );
}
