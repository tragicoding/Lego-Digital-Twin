import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { uploadAsset } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

const VIEWS = ["left", "back", "right", "front"] as const;

export default function CapturePage() {
  const { sessionId, setUploaded } = useSessionStore();
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<"character" | "object">("character");
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCharacterCapture = () => {
    setUploaded("character");
    setStep("object");
  };

  const handleObjectUpload = async () => {
    if (!sessionId) return;
    if (files.length < 4) {
      setError("front, left, back, right 4장의 사진이 필요합니다.");
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
      setError("오브제 업로드에 실패했습니다. 다시 시도해주세요.");
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10 bg-white">
      {step === "character" ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full flex flex-col items-center gap-8 text-center"
        >
          <div className="space-y-3">
            <p className="text-3xl font-black text-gray-900">캐릭터를 촬영하겠습니다.</p>
          </div>
          <PrimaryButton onClick={handleCharacterCapture}>촬영</PrimaryButton>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full flex flex-col gap-6"
        >
          <div className="text-center space-y-3">
            <p className="text-3xl font-black text-gray-900">오브제를 촬영하겠습니다.</p>
          </div>

          <button
            onClick={() => fileRef.current?.click()}
            className="aspect-square rounded-2xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center gap-3 text-gray-400 hover:border-purple-400 hover:text-purple-400 transition-colors"
          >
            <span className="text-sm font-medium">
              {files.length > 0 ? `${files.length}장 선택됨` : "4방향 사진 선택"}
            </span>
          </button>

          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []).slice(0, 4))}
          />

          {files.length > 0 && (
            <div className="grid grid-cols-2 gap-2 text-xs text-gray-500">
              {VIEWS.map((view, index) => (
                <div key={view} className="rounded-xl bg-gray-50 px-3 py-2">
                  {view}: {files[index]?.name ?? "-"}
                </div>
              ))}
            </div>
          )}

          {error && <p className="text-center text-sm text-red-500">{error}</p>}

          <PrimaryButton onClick={handleObjectUpload} disabled={uploading}>
            {uploading ? "업로드 중..." : "촬영"}
          </PrimaryButton>
        </motion.div>
      )}
    </div>
  );
}
