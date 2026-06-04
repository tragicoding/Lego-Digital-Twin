import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { uploadAsset } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

interface GuideStep  { type: "guide";   text: string }
interface CaptureStep{ type: "capture"; assetType: "character" | "object"; label: string; emoji: string; view?: "front" | "back" | "left" | "right" }
type Step = GuideStep | CaptureStep;

const STEPS: Step[] = [
  { type: "guide",   text: "안녕하세요!" },
  { type: "guide",   text: "MINIVERSE에 오신걸 환영합니다!" },
  { type: "guide",   text: "먼저 여러분의 NPC가 될\n캐릭터를 촬영할게요!" },
  { type: "guide",   text: "캐릭터를 앞에 표시된\n곳에 올려주세요!" },
  { type: "guide",   text: "캐릭터 정면 촬영을\n시작합니다" },
  { type: "capture", assetType: "character", label: "캐릭터 정면", emoji: "🧍", view: "front" },
  { type: "guide",   text: "캐릭터를 왼쪽으로\n90도 돌려주세요!" },
  { type: "guide",   text: "캐릭터 좌측면 촬영을\n시작합니다" },
  { type: "capture", assetType: "character", label: "캐릭터 좌측면", emoji: "↩️", view: "left" },
  { type: "guide",   text: "캐릭터를 다시 90도\n돌려서 뒷면을 보여주세요!" },
  { type: "guide",   text: "캐릭터 후면 촬영을\n시작합니다" },
  { type: "capture", assetType: "character", label: "캐릭터 후면", emoji: "🔄", view: "back" },
  { type: "guide",   text: "캐릭터를 다시 90도\n돌려서 우측면을 보여주세요!" },
  { type: "guide",   text: "캐릭터 우측면 촬영을\n시작합니다" },
  { type: "capture", assetType: "character", label: "캐릭터 우측면", emoji: "↪️", view: "right" },
  { type: "guide",   text: "이제 당신이 조립한\n오브제를 촬영할게요!" },
  { type: "guide",   text: "오브제를 앞에 표시된\n곳에 올려주세요!" },
  { type: "guide",   text: "오브제 촬영을\n시작합니다" },
  { type: "capture", assetType: "object", label: "오브제", emoji: "🧱" },
];

function GuideScreen({ step, onNext }: { step: GuideStep; onNext: () => void }) {
  useEffect(() => {
    const t = setTimeout(onNext, 2400);
    return () => clearTimeout(t);
  }, [step.text]);

  const words = step.text.split(/(\n)/).flatMap((seg) =>
    seg === "\n" ? ["\n"] : seg.split(" ")
  );

  return (
    <motion.div
      key={step.text}
      className="min-h-screen flex flex-col items-center justify-center px-8 cursor-pointer select-none"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onNext}
    >
      <p className="text-4xl font-black text-gray-900 text-center leading-tight">
        {words.map((w, i) =>
          w === "\n" ? (
            <br key={i} />
          ) : (
            <motion.span
              key={`${step.text}-${i}`}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07, duration: 0.35 }}
              className="inline-block mr-2"
            >
              {w}
            </motion.span>
          )
        )}
      </p>
      <motion.p
        className="text-gray-300 text-sm mt-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        화면을 탭하면 넘어갑니다
      </motion.p>
    </motion.div>
  );
}

function CaptureScreen({ step, onDone }: { step: CaptureStep; onDone: (assetId: string) => void }) {
  const { sessionId } = useSessionStore();
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setUploadError(null);
  };

  const handleUpload = () => {
    if (!sessionId || !file) return;
    setUploading(true);
    setUploadError(null);

    // 백그라운드 업로드 — await 없이 fire-and-forget
    // worker가 비동기로 처리하므로 응답을 기다릴 필요 없음
    uploadAsset(sessionId, step.assetType, file, step.view ?? "front").catch((e) => {
      console.error("백그라운드 업로드 실패", e);
    });

    // 400ms 후 즉시 다음 화면으로 이동
    setTimeout(() => onDone(""), 400);
  };

  return (
    <motion.div
      key={step.assetType}
      className="min-h-screen flex flex-col justify-center gap-5 px-6 py-10"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35 }}
    >
      <div className="text-center">
        <span className="text-5xl">{step.emoji}</span>
        <h2 className="text-2xl font-black text-gray-900 mt-2">{step.label} 촬영</h2>
      </div>

      {preview ? (
        <div className="rounded-2xl overflow-hidden aspect-square bg-gray-100">
          <img src={preview} className="w-full h-full object-cover" />
        </div>
      ) : (
        <button
          onClick={() => fileRef.current?.click()}
          className="aspect-square rounded-2xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center gap-3 text-gray-400 hover:border-purple-400 hover:text-purple-400 transition-colors"
        >
          <span className="text-5xl">📷</span>
          <span className="text-sm font-medium">사진 선택 또는 촬영</span>
        </button>
      )}

      <input ref={fileRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={handleFile} />

      {uploadError && (
        <p className="text-red-400 text-sm text-center">{uploadError}</p>
      )}

      <div className="flex gap-3">
        {preview && (
          <PrimaryButton variant="secondary" onClick={() => { setPreview(null); setFile(null); }}>
            다시 찍기
          </PrimaryButton>
        )}
        <PrimaryButton onClick={preview ? handleUpload : () => fileRef.current?.click()} disabled={uploading}>
          {uploading ? "업로드 중..." : preview ? "업로드" : "촬영하기"}
        </PrimaryButton>
      </div>
    </motion.div>
  );
}

export default function CapturePage() {
  const [stepIdx, setStepIdx] = useState(0);
  const navigate = useNavigate();
  const { setUploaded, setCharacterAssetId, setObjectAssetId } = useSessionStore();

  const step = STEPS[stepIdx];
  const next = () => setStepIdx((i) => i + 1);

  const handleCaptureDone = (assetId: string) => {
    const s = step as CaptureStep;
    if (s.assetType === "character" && (s.view ?? "front") === "front") {
      setCharacterAssetId(assetId);
      setUploaded("character");
    } else if (s.assetType === "object") {
      setObjectAssetId(assetId);
      setUploaded("object");
    }
    if (stepIdx >= STEPS.length - 1) navigate("/profile");
    else next();
  };

  const captureSteps = STEPS.filter((s) => s.type === "capture");
  const doneCount = captureSteps.filter((s) => STEPS.indexOf(s) < stepIdx).length;

  return (
    <div className="relative">
      <div className="fixed top-0 left-0 right-0 z-10 flex gap-2 px-6 pt-5 max-w-md mx-auto">
        {captureSteps.map((_, i) => (
          <div key={i} className={`flex-1 h-1 rounded-full transition-colors duration-500 ${i < doneCount ? "bg-purple-500" : "bg-gray-200"}`} />
        ))}
      </div>
      <AnimatePresence mode="wait">
        {step.type === "guide" ? (
          <GuideScreen key={stepIdx} step={step} onNext={next} />
        ) : (
          <CaptureScreen key={stepIdx} step={step as CaptureStep} onDone={handleCaptureDone} />
        )}
      </AnimatePresence>
    </div>
  );
}
