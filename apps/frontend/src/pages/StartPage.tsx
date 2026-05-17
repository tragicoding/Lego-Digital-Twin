/*
session 생성
StartPage 접속
↓
사용자가 시작하기 클릭
↓
createSession()으로 백엔드에 세션 생성 요청
↓
응답으로 session_id 받음
↓
Zustand store에 session_id 저장
↓
/capture 페이지로 이동

*/



import { useNavigate } from "react-router-dom";
import { createSession } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

export default function StartPage() {
  //페이지 이동을 위함.
  const navigate = useNavigate();
  const setSessionId = useSessionStore((s) => s.setSessionId);
//시작하기 버튼 클릭 시, 서버에 세션 생성 요청을 보내고
// , 응답으로 받은 session_id를 전역 상태에 저장한 뒤,
//  촬영 페이지로 이동한다.
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
