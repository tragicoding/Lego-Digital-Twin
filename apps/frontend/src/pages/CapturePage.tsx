/*
캐릭터 → 건축물 → 자동차 순서로 사진을 선택/촬영하고, 
백엔드에 업로드하는 페이지
*/




import { useState, useRef } from "react";
//useState: 화면 상태 관리
//useRef: 숨겨진 file input을 버튼으로 클릭시키기 위해 사용
import { useNavigate } from "react-router-dom";
import { uploadAsset } from "../api/client";
//백엔드에 이미지 업로드하는 API 함수
import { useSessionStore } from "../store/sessionStore";
//sessionId, 업로드 완료 상태를 가져오기 위해 사용


const STEPS = [
  { type: "character", label: "캐릭터", emoji: "🧍", desc: "레고 캐릭터를 조립하고 촬영하세요" },
  { type: "building",  label: "건축물", emoji: "🏰", desc: "레고 건축물을 조립하고 촬영하세요" },
  { type: "vehicle",   label: "자동차", emoji: "🚗", desc: "레고 자동차를 조립하고 촬영하세요" },
]; //촬영 단계 정의 배열 

export default function CapturePage() {
  const navigate = useNavigate();
  const { sessionId, setUploaded, uploads } = useSessionStore();
  //전역 store에서 세 가지를 가져온다.
  /*
  {
  sessionId: "abc123",
  uploads: {
    character: true,
    building: true
  }
}
  */
  const [current, setCurrent] = useState(0); //useState 상태들
  const [loading, setLoading] = useState(false);//업로드 중인지 여부
  const [preview, setPreview] = useState<string | null>(null); //선택한 이미지 미리보기 URL을 저장
  //사진을 선택하면 화면에 이미지가 보이게 만드는 값
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
//숨겨진 파일 선택 input을 직접 클릭시키기 위한 참조.
//실제 input은 화면에 안 보이게 되어 있음.
  const step = STEPS[current];//현재 단계 정보 가져옴

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!sessionId || !selectedFile) return;//세션 ID가 없거나 파일이 없으면 업로드하지 않고 종료

    setLoading(true); //업로드 시작 상태로 변경
    await uploadAsset(sessionId, step.type, selectedFile);
    //백엔드에 이미지 업로드 요청 보낸다.
    //sessionID, step.type,selectedFile을 보낸다. 
    setUploaded(step.type);//업로드 끝나면 Zustand에 기록

    //초기화 
    setPreview(null);
    setSelectedFile(null);
    setLoading(false);

    //다음 단계로 이동. 
    //현재 단계가 마지막 단계가 아니면 다음 단계로 이동, 마지막 단계면 프로필 페이지로 이동
    // -> /profile 페이지로 이동.
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
