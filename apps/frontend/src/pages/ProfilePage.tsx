import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { updateProfile } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

export default function ProfilePage() {
  const navigate = useNavigate();
  const sessionId = useSessionStore((s) => s.sessionId);
  const [nickname, setNickname] = useState("");
  const [bubbleText, setBubbleText] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!sessionId || !nickname.trim()) return;
    setLoading(true);
    await updateProfile(sessionId, { nickname, bubble_text: bubbleText });
    navigate("/loading");
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-md bg-white rounded-3xl shadow-lg p-8 flex flex-col gap-6">
        <div className="text-center">
          <span className="text-5xl">✏️</span>
          <h2 className="text-2xl font-bold text-gray-900 mt-3">내 정보 입력</h2>
          <p className="text-gray-500 text-sm mt-2">
            입력한 이름이 VR 속 캐릭터 이름이 됩니다
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700 mb-1 block">
              닉네임 <span className="text-blue-500">*</span>
            </label>
            <input
              type="text"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              placeholder="VR 속 내 이름"
              maxLength={20}
              className="w-full px-4 py-3 rounded-2xl border border-gray-200 focus:border-blue-400 focus:outline-none text-gray-900"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700 mb-1 block">
              말풍선 문구
            </label>
            <input
              type="text"
              value={bubbleText}
              onChange={(e) => setBubbleText(e.target.value)}
              placeholder="다른 관객에게 보여줄 한 마디"
              maxLength={50}
              className="w-full px-4 py-3 rounded-2xl border border-gray-200 focus:border-blue-400 focus:outline-none text-gray-900"
            />
            <p className="text-xs text-gray-400 mt-1">
              캐릭터 NPC가 다른 관객에게 이 문구를 보여줍니다
            </p>
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={!nickname.trim() || loading}
          className="w-full py-4 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-2xl text-lg transition-colors disabled:opacity-40"
        >
          {loading ? "저장 중..." : "완료"}
        </button>
      </div>
    </div>
  );
}
