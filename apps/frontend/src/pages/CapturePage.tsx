import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { captureObjectFromCameras, prepareCharacterPlaceholder } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

const shutterSound = new Audio("/sounds/camera-shutter.mp3");
shutterSound.volume = 0.7;

function playShutterSound() {
  shutterSound.currentTime = 0;
  shutterSound.play().catch(() => {});
}

interface CapturePageProps {
  testMode?: boolean;
}

export default function CapturePage({ testMode = false }: CapturePageProps) {
  const { sessionId, setUploaded } = useSessionStore();
  const navigate = useNavigate();
  const [step, setStep] = useState<"character" | "object">("character");
  const [uploading, setUploading] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const runCountdown = async () => {
    setError(null);
    for (const value of [3, 2, 1]) {
      setCountdown(value);
      await delay(850);
    }
    setCountdown(null);
  };

  const handleCharacterCapture = async () => {
    if (uploading) return;
    playShutterSound();
    setUploading(true);
    await runCountdown();

    if (testMode) {
      setUploaded("character");
      setStep("object");
      setUploading(false);
      return;
    }

    if (!sessionId) {
      setError("세션 정보가 없습니다. 처음부터 다시 시작해 주세요.");
      setUploading(false);
      return;
    }

    try {
      await prepareCharacterPlaceholder(sessionId);
      setUploaded("character");
      setStep("object");
    } catch {
      setError("캐릭터 세션 준비에 실패했습니다. 다시 시도해 주세요.");
    } finally {
      setUploading(false);
    }
  };

  const handleObjectCapture = async () => {
    if (uploading) return;
    playShutterSound();
    setUploading(true);
    await runCountdown();

    if (testMode) {
      setUploaded("object");
      navigate("/test/loading");
      return;
    }

    if (!sessionId) {
      setError("세션 정보가 없습니다. 처음부터 다시 시작해 주세요.");
      setUploading(false);
      return;
    }

    try {
      await captureObjectFromCameras(sessionId);
      setUploaded("object");
      navigate("/loading");
    } catch {
      setError("오브제 카메라 촬영에 실패했습니다. Windows 카메라 서버를 확인해 주세요.");
      setUploading(false);
    }
  };

  const renderCaptureStep = (
    title: string,
    onCapture: () => void,
    disabled = false,
    buttonLabel = "촬영",
  ) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full flex flex-col items-center gap-6 text-center"
    >
      <motion.img
        src="/images/camera.png"
        alt=""
        className="w-52 max-w-[68vw] object-contain"
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35 }}
      />
      <div className="space-y-3">
        <p className="text-3xl font-black text-gray-900">{title}</p>
      </div>
      <PrimaryButton onClick={onCapture} disabled={disabled}>
        {buttonLabel}
      </PrimaryButton>
      {error && <p className="text-center text-sm text-red-500">{error}</p>}
    </motion.div>
  );

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10 bg-white">
      {countdown !== null ? (
        <motion.div
          key={countdown}
          className="flex h-screen w-full items-center justify-center bg-white"
          initial={{ opacity: 0, scale: 0.45, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 1.35 }}
          transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
        >
          <motion.span
            className="text-[9rem] font-black leading-none text-gray-950"
            animate={{ scale: [1, 1.08, 1], opacity: [0.85, 1, 0.9] }}
            transition={{ duration: 0.65, ease: "easeOut" }}
          >
            {countdown}
          </motion.span>
        </motion.div>
      ) : step === "character" ? (
        renderCaptureStep(
          "캐릭터 촬영하기",
          handleCharacterCapture,
          uploading,
          uploading ? "촬영 중..." : "촬영하기",
        )
      ) : (
        renderCaptureStep(
          "오브제 촬영하기",
          handleObjectCapture,
          uploading,
          uploading ? "촬영 중..." : "촬영하기",
        )
      )}
    </div>
  );
}
