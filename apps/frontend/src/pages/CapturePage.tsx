import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { uploadAsset } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

const VIEWS = ["front", "left", "back", "right"] as const;

interface CapturePageProps {
  testMode?: boolean;
}

export default function CapturePage({ testMode = false }: CapturePageProps) {
  const { sessionId, setUploaded } = useSessionStore();
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<"character" | "object">("character");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCharacterCapture = () => {
    setUploaded("character");
    setStep("object");
  };

  const handleObjectCapture = () => {
    if (testMode) {
      setUploaded("object");
      navigate("/test/loading");
      return;
    }

    fileRef.current?.click();
  };

  const handleObjectFiles = async (selectedFiles: File[]) => {
    if (!sessionId) return;

    const files = selectedFiles.slice(0, 4);
    if (files.length < 4) {
      setError("front, left, back, right 사진 4장이 필요합니다.");
      return;
    }

    setUploading(true);
    setError(null);
    try {
      for (let i = 0; i < VIEWS.length; i += 1) {
        await uploadAsset(sessionId, "object", files[i], VIEWS[i]);
      }
      setUploaded("object");
      navigate("/loading");
    } catch {
      setError("오브제 업로드에 실패했습니다. 다시 시도해 주세요.");
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
      className="w-full flex flex-col items-center gap-8 text-center"
    >
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
      {step === "character"
        ? renderCaptureStep("캐릭터를 촬영하겠습니다", handleCharacterCapture)
        : renderCaptureStep(
            "오브제를 촬영하겠습니다",
            handleObjectCapture,
            uploading,
            uploading ? "업로드 중..." : "촬영",
          )}

      {!testMode && (
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => void handleObjectFiles(Array.from(e.target.files ?? []))}
        />
      )}
    </div>
  );
}
