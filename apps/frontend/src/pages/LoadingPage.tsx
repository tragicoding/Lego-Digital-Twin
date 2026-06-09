import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { finalizeSession, getStatus } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

const STAGE_MESSAGES: Record<string, string> = {
  waiting:    "디지털 변환을 준비하고 있어요",
  queued:     "디지털 변환을 준비하고 있어요",
  generating: "레고 모형을 3D 모델로 만들고 있어요",
  rigging:    "캐릭터에 움직임을 부여하고 있어요",
  animating:  "캐릭터에 움직임을 부여하고 있어요",
  downloading:"MINIVERSE 월드로 이동 중이에요",
  ready:      "준비가 완료되었어요",
  completed:  "준비가 완료되었어요",
};

interface AssetStatus { status: string; stage: string; progress: number }

const ITEMS = [
  { key: "character", label: "캐릭터", emoji: "🧍" },
  { key: "object",    label: "오브제", emoji: "🧱" },
];

function ProgressCard({ label, emoji, asset }: { label: string; emoji: string; asset?: AssetStatus }) {
  const progress = asset?.progress ?? 0;
  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-2xl">{emoji}</span>
        <span className="font-bold text-gray-800">{label}</span>
        {asset?.status === "completed" && <span className="ml-auto text-green-500 text-sm font-semibold">완료 ✓</span>}
        {asset?.status === "failed"    && <span className="ml-auto text-red-400 text-sm font-semibold">오류</span>}
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
        <motion.div
          className="h-2 rounded-full"
          style={{ background: "linear-gradient(90deg, #7C3AED, #38BDF8)" }}
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.6 }}
        />
      </div>
      <p className="text-xs text-gray-400 mt-2">
        {STAGE_MESSAGES[asset?.stage ?? "waiting"] ?? "대기 중"}
      </p>
    </div>
  );
}

export default function LoadingPage() {
  const { sessionId, reset } = useSessionStore();
  const navigate = useNavigate();
  const [assets, setAssets] = useState<Record<string, AssetStatus>>({});
  const [readyForUnity, setReadyForUnity] = useState(false);
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    const poll = setInterval(async () => {
      try {
        const res = await getStatus(sessionId);
        setAssets(res.data.assets);
        setReadyForUnity(res.data.ready_for_unity);
        if (res.data.ready_for_unity) clearInterval(poll);
      } catch {
        // 네트워크 일시 오류는 무시하고 다음 폴링에서 재시도
      }
    }, 3000);
    return () => clearInterval(poll);
  }, [sessionId]);

  const getAsset = (key: string) =>
    assets[key] ?? (key === "object" ? assets["building"] : undefined);

  const handleRestart = async () => {
    if (restarting) return;
    setRestarting(true);
    if (sessionId) {
      try {
        await finalizeSession(sessionId);
      } catch (error) {
        console.error("세션 finalize 실패", error);
      }
    }
    reset();
    navigate("/start");
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10">
      <div className="w-full flex flex-col gap-5">
        <AnimatePresence mode="wait">
          {!readyForUnity ? (
            <motion.div key="loading" className="flex flex-col gap-5">
              <div className="text-center">
                <h2 className="text-xl font-black text-gray-900 leading-snug">
                  3D로 변환 중입니다!
                </h2>
                <p className="text-gray-400 text-sm mt-2">잠시만 기다려 주세요</p>
              </div>
              {ITEMS.map(({ key, label, emoji }) => (
                <ProgressCard key={key} label={label} emoji={emoji} asset={getAsset(key)} />
              ))}
            </motion.div>
          ) : (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center gap-8 text-center"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 260, damping: 14 }}
                className="text-7xl"
              >
                🎉
              </motion.div>
              <div className="space-y-2">
                <h2 className="text-2xl font-black text-gray-900">완료되었습니다!</h2>
                <p className="text-gray-500 text-base leading-relaxed">이제 VR World로 이동할게요!</p>
              </div>
              <div
                className="w-full py-5 rounded-2xl text-white font-bold text-base text-center"
                style={{ background: "linear-gradient(135deg, #7C3AED, #38BDF8)" }}
              >
                VR 입장 준비 완료 ✓
              </div>
              <button
                onClick={handleRestart}
                disabled={restarting}
                className="w-full py-4 rounded-2xl border-2 border-gray-200 text-gray-500 font-bold text-base"
              >
                {restarting ? "준비 중..." : "새롭게 시작하기"}
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
