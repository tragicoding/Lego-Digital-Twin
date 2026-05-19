import { useEffect, useState } from "react";
import { getStatus } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

const STAGE_MESSAGES: Record<string, string> = {
  waiting:    "디지털 변환을 준비하고 있어요",
  generating: "레고 모형을 3D 모델로 만들고 있어요",
  rigging:    "캐릭터에 움직임을 부여하고 있어요",
  animating:  "캐릭터가 걸을 준비를 하고 있어요",
  downloading:"Unity 월드로 이동 중이에요",
  ready:      "준비 완료!",
};

interface AssetStatus {
  status: string;
  stage: string;
  progress: number;
}

export default function LoadingPage() {
  const sessionId = useSessionStore((s) => s.sessionId);
  const [assets, setAssets] = useState<Record<string, AssetStatus>>({});
  const [readyForUnity, setReadyForUnity] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(async () => {
      const res = await getStatus(sessionId);
      setAssets(res.data.assets);
      setReadyForUnity(res.data.ready_for_unity);
      if (res.data.ready_for_unity) clearInterval(interval);
    }, 3000);
    return () => clearInterval(interval);
  }, [sessionId]);

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
