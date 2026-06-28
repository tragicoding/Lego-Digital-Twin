import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { getStatus } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

interface LoadingPageProps {
  testMode?: boolean;
}

export default function LoadingPage({ testMode = false }: LoadingPageProps) {
  const { sessionId, reset } = useSessionStore();
  const navigate = useNavigate();
  const [ready, setReady] = useState(false);
  const [progress, setProgress] = useState(0);
  const [moving, setMoving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (testMode) {
      const timer = setTimeout(() => setReady(true), 1200);
      return () => clearTimeout(timer);
    }

    if (!sessionId) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const res = await getStatus(sessionId);
        if (cancelled) return;

        const assets = Object.values(res.data.assets ?? {}) as Array<{ progress?: number }>;
        const avg = assets.length
          ? Math.round(assets.reduce((sum, item) => sum + (item.progress ?? 0), 0) / assets.length)
          : 0;
        setProgress(avg);

        if (res.data.ready_for_unity) {
          setReady(true);
          setError(null);
        }
      } catch {
        if (!cancelled) setError("처리 상태를 불러오지 못했습니다.");
      }
    };

    void poll();
    const timer = setInterval(() => void poll(), 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [sessionId, testMode]);

  const handleMove = async () => {
    if (moving) return;
    setMoving(true);
    setError(null);

    if (testMode) {
      reset();
      navigate("/test/start");
      return;
    }

    reset();
    navigate("/start");
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-8 bg-white text-center">
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        <p className="text-3xl font-black leading-snug text-gray-900 whitespace-pre-line">
          {ready
            ? "MINIVERSE로 이동할게요\n앞의 부스로 이동하세요"
            : `3D 모델을 만들고 있어요\n${progress}%`}
        </p>

        {ready && (
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
