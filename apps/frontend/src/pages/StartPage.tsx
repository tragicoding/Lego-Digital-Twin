import { useNavigate } from "react-router-dom";
import { createSession } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

export default function StartPage() {
  const navigate = useNavigate();
  const setSessionId = useSessionStore((s) => s.setSessionId);

  const handleStart = async () => {
    const res = await createSession();
    setSessionId(res.data.session_id);
    navigate("/capture");
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-md bg-white rounded-3xl shadow-lg p-10 flex flex-col items-center gap-8">
        {/* 로고 */}
        <div className="w-16 h-16 bg-blue-500 rounded-2xl flex items-center justify-center">
          <span className="text-white text-3xl">🧱</span>
        </div>

        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-3">레고 VR 월드</h1>
          <p className="text-gray-500 text-base leading-relaxed">
            직접 만든 레고가<br />
            VR 놀이공원 안에 등장합니다.
          </p>
        </div>

        <div className="w-full bg-blue-50 rounded-2xl p-5">
          <p className="text-blue-700 text-sm font-medium text-center">
            촬영 순서 안내
          </p>
          <div className="mt-3 flex justify-around">
            {["캐릭터", "건축물", "자동차"].map((item, i) => (
              <div key={item} className="flex flex-col items-center gap-1">
                <span className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                  {i + 1}
                </span>
                <span className="text-gray-700 text-sm">{item}</span>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={handleStart}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-4 rounded-2xl text-lg transition-colors"
        >
          시작하기
        </button>
      </div>
    </div>
  );
}
