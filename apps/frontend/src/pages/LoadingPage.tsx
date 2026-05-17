/*
업로드된 레고 이미지들이 3D 모델로 변환되는 상태를
주기적으로 확인하고, 진행률을 화면에 보여주는 page.

LoadingPage 진입
↓
Zustand에서 sessionId 가져옴
↓
3초마다 getStatus(sessionId) 호출
↓
백엔드에서 assets 상태와 ready_for_unity 값 받음
↓
캐릭터/건축물/자동차 진행률 표시
↓
모든 변환이 완료되면 readyForUnity = true
↓
반복 조회 중단
↓
VR 입장 준비 완료 표시

*/

//useState: 화면 상태 저장
//useEffect: 페이지가 실행될 때 특정 작업 수행(백엔드상태를 3초마다 확인하는 기능.)
import { useEffect, useState } from "react";
//백엔드에서 현재 세션의 처리 상태를 가져오는 API 함수
import { getStatus } from "../api/client";
//전역 상태에서 sessionId를 가져옴.
import { useSessionStore } from "../store/sessionStore";



//백엔드에서 오는 stage 값을 사용자에게 보여줄 문장으로 바꿔주는 객체
const STAGE_MESSAGES: Record<string, string> = {
  waiting:    "디지털 변환을 준비하고 있어요",
  generating: "레고 모형을 3D 모델로 만들고 있어요",
  rigging:    "캐릭터에 움직임을 부여하고 있어요",
  animating:  "캐릭터가 걸을 준비를 하고 있어요",
  downloading:"Unity 월드로 이동 중이에요",
  ready:      "준비 완료!",
};


//각 asset 타입의 상태 타입 정의. 
interface AssetStatus {
  status: string;
  stage: string;
  progress: number;
}

/*
{
  status: "processing",
  stage: "generating",
  progress: 45
}
*/

export default function LoadingPage() {
  const sessionId = useSessionStore((s) => s.sessionId);//현재 세션ID가져온다.
  const [assets, setAssets] = useState<Record<string, AssetStatus>>({});//각 모델의 변환 상태 저장
  /*
  {
  character: {
    status: "completed",
    stage: "ready",
    progress: 100
  },
  building: {
    status: "processing",
    stage: "generating",
    progress: 60
  },
  vehicle: {
    status: "waiting",
    stage: "waiting",
    progress: 0
  }
}
  */
  const [readyForUnity, setReadyForUnity] = useState(false);
  //Unity로 이동할 준비가 되었는지 여부 저장

  //sessionId가 없으면 상태 조회를 하지 않고 종료
  useEffect(() => {
    if (!sessionId) return;
    //일정 시간마다 함수를 반복 실행(setinterval)
    const interval = setInterval(async () => {
      const res = await getStatus(sessionId);
      setAssets(res.data.assets);
      setReadyForUnity(res.data.ready_for_unity);
      if (res.data.ready_for_unity) clearInterval(interval);
    }, 3000);
    /*
    3초마다 getStatus(sessionId) 호출
→ 백엔드에서 현재 변환 상태 받음
→ assets 상태 업데이트
→ ready_for_unity 값 업데이트
→ 준비 완료되면 반복 조회 중단
    */
   //cleanup 함수: 컴포넌트가 사라질 때 인터벌도 같이 정리
    return () => clearInterval(interval);
  }, [sessionId]);


  //화면에 표시할 항목 목록
  const ITEMS = [
    { key: "character", label: "캐릭터", emoji: "🧍" },
    { key: "building",  label: "건축물", emoji: "🏰" },
    { key: "vehicle",   label: "자동차", emoji: "🚗" },
  ];

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-md flex flex-col gap-5">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900">
            {readyForUnity ? "🎉 VR 월드 준비 완료!" : "3D 변환 중..."}
          </h2>
          <p className="text-gray-500 text-sm mt-2">
            {readyForUnity
              ? "VR 헤드셋을 착용하고 입장하세요"
              : "잠시만 기다려 주세요"}
          </p>
        </div>

        {ITEMS.map(({ key, label, emoji }) => {
          const a = assets[key];
          return (
            <div key={key} className="bg-white rounded-2xl shadow p-5">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-2xl">{emoji}</span>
                <span className="font-semibold text-gray-800">{label}</span>
                {a?.status === "completed" && (
                  <span className="ml-auto text-green-500 text-sm font-medium">완료</span>
                )}
                {a?.status === "failed" && (
                  <span className="ml-auto text-red-400 text-sm font-medium">오류</span>
                )}
              </div>
              <div className="w-full bg-gray-100 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${a?.progress ?? 0}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {STAGE_MESSAGES[a?.stage ?? "waiting"] ?? "대기 중"}
              </p>
            </div>
          );
        })}

        {readyForUnity && (
          <div className="bg-blue-500 rounded-2xl p-6 text-center text-white">
            <p className="text-lg font-bold">VR 입장 준비 완료 ✓</p>
            <p className="text-sm opacity-80 mt-1">Unity VR PC에서 세션을 시작하세요</p>
          </div>
        )}
      </div>
    </div>
  );
}
