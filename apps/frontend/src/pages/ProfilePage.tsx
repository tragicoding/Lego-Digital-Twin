import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { updateProfile } from "../api/client";
import { useSessionStore } from "../store/sessionStore";
import PrimaryButton from "../components/PrimaryButton";

interface NameStepProps {
  title: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  onNext: () => void;
  buttonLabel: string;
  disabled: boolean;
}

function NameStep({ title, placeholder, value, onChange, onNext, buttonLabel, disabled }: NameStepProps) {
  return (
    <motion.div
      className="min-h-screen flex flex-col items-center justify-center gap-8 px-8"
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -40 }}
      transition={{ duration: 0.32 }}
    >
      <p className="text-3xl font-black text-gray-900 text-center leading-snug whitespace-pre-line">
        {title}
      </p>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        maxLength={20}
        autoFocus
        className="w-full px-5 py-4 rounded-2xl border-2 border-gray-200 focus:border-purple-400 focus:outline-none text-gray-900 text-lg font-semibold text-center"
      />
      <div className="w-full">
        <PrimaryButton onClick={onNext} disabled={disabled}>
          {buttonLabel}
        </PrimaryButton>
      </div>
    </motion.div>
  );
}

export default function ProfilePage() {
  const navigate = useNavigate();
  const { sessionId, setCharacterNpcName, setObjectName } = useSessionStore();
  const [step, setStep] = useState<0 | 1>(0);
  const [charName, setCharName] = useState("");
  const [objName, setObjName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!sessionId || !objName.trim()) return;
    setSubmitting(true);
    setCharacterNpcName(charName);
    setObjectName(objName);
    await updateProfile(sessionId, { character_npc_name: charName, object_name: objName });
    navigate("/loading");
  };

  return (
    <div className="relative">
      <div className="fixed top-6 left-0 right-0 z-10 flex justify-center gap-2 max-w-md mx-auto px-8">
        {[0, 1].map((i) => (
          <motion.div
            key={i}
            animate={{ width: i === step ? 24 : 8 }}
            className={`h-2 rounded-full transition-colors duration-300 ${i === step ? "bg-purple-500" : "bg-gray-200"}`}
          />
        ))}
      </div>
      <AnimatePresence mode="wait">
        {step === 0 ? (
          <NameStep
            key="char"
            title={"캐릭터 NPC의\n이름을 지어주세요!"}
            placeholder="예) 몽글이"
            value={charName}
            onChange={setCharName}
            onNext={() => setStep(1)}
            buttonLabel="다음"
            disabled={!charName.trim()}
          />
        ) : (
          <NameStep
            key="obj"
            title={"오브제의\n이름을 지어주세요!"}
            placeholder="예) 하늘탑"
            value={objName}
            onChange={setObjName}
            onNext={handleSubmit}
            buttonLabel={submitting ? "저장 중..." : "완료"}
            disabled={!objName.trim() || submitting}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
