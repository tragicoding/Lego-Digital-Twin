import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { uploadAsset } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

const STEPS = [
  { type: "character", label: "캐릭터", emoji: "🧍", desc: "레고 캐릭터를 조립하고 촬영하세요" },
  { type: "building",  label: "건축물", emoji: "🏰", desc: "레고 건축물을 조립하고 촬영하세요" },
  { type: "vehicle",   label: "자동차", emoji: "🚗", desc: "레고 자동차를 조립하고 촬영하세요" },
];

export default function CapturePage() {
  const navigate = useNavigate();
  const { sessionId, setUploaded, uploads } = useSessionStore();
  const [current, setCurrent] = useState(0);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const step = STEPS[current];

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!sessionId || !selectedFile) return;
    setLoading(true);
    await uploadAsset(sessionId, step.type, selectedFile);
    setUploaded(step.type);
    setPreview(null);
    setSelectedFile(null);
    setLoading(false);

    if (current < STEPS.length - 1) {
      setCurrent((c) => c + 1);
    } else {
      navigate("/profile");
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-md flex flex-col gap-5">
        {/* 진행 바 */}
        <div className="flex gap-2">
          {STEPS.map((s, i) => (
            <div
              key={s.type}
              className={`flex-1 h-1.5 rounded-full transition-colors ${
                i <= current ? "bg-blue-500" : "bg-gray-200"
              }`}
            />
          ))}
        </div>

        <div className="bg-white rounded-3xl shadow-lg p-8 flex flex-col gap-6">
          <div className="text-center">
            <span className="text-5xl">{step.emoji}</span>
            <h2 className="text-2xl font-bold text-gray-900 mt-3">{step.label} 촬영</h2>
            <p className="text-gray-500 text-sm mt-2">{step.desc}</p>
          </div>

          {/* 미리보기 */}
          {preview ? (
            <div className="relative rounded-2xl overflow-hidden bg-gray-100 aspect-square">
              <img src={preview} className="w-full h-full object-cover" />
            </div>
          ) : (
            <button
              onClick={() => fileRef.current?.click()}
              className="aspect-square rounded-2xl border-2 border-dashed border-gray-300 flex flex-col items-center justify-center gap-3 text-gray-400 hover:border-blue-400 hover:text-blue-400 transition-colors"
            >
              <span className="text-4xl">📷</span>
              <span className="text-sm font-medium">사진 선택 또는 촬영</span>
            </button>
          )}

          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFile}
          />

          <div className="flex gap-3">
            {preview && (
              <button
                onClick={() => { setPreview(null); setSelectedFile(null); }}
                className="flex-1 py-3 rounded-2xl border border-gray-200 text-gray-600 font-medium"
              >
                다시 찍기
              </button>
            )}
            <button
              onClick={preview ? handleUpload : () => fileRef.current?.click()}
              disabled={loading}
              className="flex-1 py-3 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-2xl transition-colors disabled:opacity-50"
            >
              {loading ? "업로드 중..." : preview ? "업로드" : "촬영하기"}
            </button>
          </div>
        </div>

        {/* 완료 항목 */}
        <div className="flex gap-2 justify-center">
          {STEPS.map((s) => (
            <div
              key={s.type}
              className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium ${
                uploads[s.type]
                  ? "bg-blue-100 text-blue-600"
                  : "bg-gray-100 text-gray-400"
              }`}
            >
              {uploads[s.type] ? "✓" : "○"} {s.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
