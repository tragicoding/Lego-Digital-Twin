/*
  테스트 전용 페이지.
  실제 사진 없이 Mock 데이터로 전체 파이프라인(세션→업로드→처리→Unity 준비)을 테스트한다.
*/
import { useState, useEffect } from "react";
import { api, getStatus } from "../api/client";
import { useSessionStore } from "../store/sessionStore";

interface AssetStatus {
  status: string;
  stage: string;
  progress: number;
  model_url: string | null;
}

const STAGE_MSG: Record<string, string> = {
  waiting:    "대기 중",
  generating: "3D 모델 생성 중",
  rigging:    "리깅 중 (캐릭터)",
  animating:  "애니메이션 적용 중",
  downloading:"다운로드 중",
  ready:      "완료",
};

const ITEMS = [
  { key: "character", label: "캐릭터", emoji: "🧍" },
  { key: "building",  label: "건물",   emoji: "🏰" },
  { key: "vehicle",   label: "자동차", emoji: "🚗" },
];

export default function TestPage() {
  const { sessionId, setSessionId } = useSessionStore();
  const [loading, setLoading]   = useState(false);
  const [assets, setAssets]     = useState<Record<string, AssetStatus>>({});
  const [ready, setReady]       = useState(false);
  const [unityData, setUnityData] = useState<any>(null);
  const [nickname, setNickname] = useState("테스트유저");
  const [bubble, setBubble]     = useState("안녕하세요! 제 레고 월드에 오신 걸 환영해요 🧱");
  const [log, setLog]           = useState<string[]>([]);

  const addLog = (msg: string) =>
    setLog((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 30));

  // 상태 polling
  useEffect(() => {
    if (!sessionId || ready) return;
    const id = setInterval(async () => {
      const res = await getStatus(sessionId);
      setAssets(res.data.assets);
      setReady(res.data.ready_for_unity);
      if (res.data.ready_for_unity) {
        addLog("✅ ready_for_unity = true");
        clearInterval(id);
        fetchUnityData(sessionId);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [sessionId, ready]);

  const fetchUnityData = async (sid: string) => {
    const res = await api.get(`/unity/sessions/${sid}`);
    setUnityData(res.data);
    addLog("Unity 데이터 수신 완료");
  };

  const handleStart = async () => {
    setLoading(true);
    setAssets({});
    setReady(false);
    setUnityData(null);
    setLog([]);

    addLog("세션 + Mock 파이프라인 시작...");
    const res = await api.post(
      `/test/start?nickname=${encodeURIComponent(nickname)}&bubble_text=${encodeURIComponent(bubble)}`
    );
    const sid = res.data.session_id;
    setSessionId(sid);
    addLog(`세션 생성: ${sid}`);
    addLog(`asset_ids: ${JSON.stringify(res.data.asset_ids)}`);
    setLoading(false);
  };

  const handleReset = async () => {
    if (sessionId) {
      await api.delete(`/test/sessions/${sessionId}`);
      addLog(`세션 삭제: ${sessionId}`);
    }
    setSessionId("");
    setAssets({});
    setReady(false);
    setUnityData(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto flex flex-col gap-5">

        {/* 헤더 */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-4">
          <p className="text-yellow-800 font-bold text-sm">🧪 테스트 모드</p>
          <p className="text-yellow-700 text-xs mt-1">
            실제 사진 없이 Mock 데이터로 전체 파이프라인을 테스트합니다
          </p>
        </div>

        {/* 입력 */}
        <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-4">
          <h2 className="font-bold text-gray-800">테스트 설정</h2>
          <input
            className="border border-gray-200 rounded-xl px-4 py-2 text-sm"
            value={nickname}
            onChange={(e) => setNickname(e.target.value)}
            placeholder="닉네임"
          />
          <input
            className="border border-gray-200 rounded-xl px-4 py-2 text-sm"
            value={bubble}
            onChange={(e) => setBubble(e.target.value)}
            placeholder="말풍선 문구"
          />
          <div className="flex gap-3">
            <button
              onClick={handleStart}
              disabled={loading}
              className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 rounded-xl disabled:opacity-50"
            >
              {loading ? "시작 중..." : "🚀 Mock 테스트 시작"}
            </button>
            <button
              onClick={handleReset}
              className="px-4 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-xl text-sm"
            >
              초기화
            </button>
          </div>
        </div>

        {/* 세션 정보 */}
        {sessionId && (
          <div className="bg-white rounded-2xl shadow p-4">
            <p className="text-xs text-gray-500">세션 ID</p>
            <p className="font-mono text-sm text-blue-600">{sessionId}</p>
          </div>
        )}

        {/* 진행 상태 */}
        {sessionId && (
          <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-4">
            <h2 className="font-bold text-gray-800">처리 상태</h2>
            {ITEMS.map(({ key, label, emoji }) => {
              const a = assets[key];
              return (
                <div key={key}>
                  <div className="flex items-center gap-2 mb-1">
                    <span>{emoji}</span>
                    <span className="text-sm font-medium text-gray-700">{label}</span>
                    <span className="ml-auto text-xs text-gray-400">
                      {a ? `${a.progress}% · ${STAGE_MSG[a.stage] ?? a.stage}` : "대기"}
                    </span>
                    {a?.status === "completed" && <span className="text-green-500 text-xs">✓</span>}
                    {a?.status === "failed"    && <span className="text-red-400 text-xs">✗</span>}
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-700 ${
                        a?.status === "failed" ? "bg-red-400" : "bg-blue-500"
                      }`}
                      style={{ width: `${a?.progress ?? 0}%` }}
                    />
                  </div>
                  {a?.model_url && (
                    <p className="text-xs text-gray-400 mt-0.5 truncate">{a.model_url}</p>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Unity 준비 완료 */}
        {ready && (
          <div className="bg-blue-500 rounded-2xl p-5 text-white text-center">
            <p className="text-xl font-bold">🎉 Unity 입장 준비 완료!</p>
            <p className="text-sm opacity-80 mt-1">ready_for_unity = true</p>
          </div>
        )}

        {/* Unity 데이터 */}
        {unityData && (
          <div className="bg-white rounded-2xl shadow p-4">
            <p className="font-bold text-gray-800 mb-2">Unity 수신 데이터</p>
            <pre className="text-xs text-gray-600 overflow-auto bg-gray-50 rounded-xl p-3 max-h-64">
              {JSON.stringify(unityData, null, 2)}
            </pre>
          </div>
        )}

        {/* 로그 */}
        {log.length > 0 && (
          <div className="bg-white rounded-2xl shadow p-4">
            <p className="font-bold text-gray-800 mb-2 text-sm">이벤트 로그</p>
            <div className="flex flex-col gap-1">
              {log.map((l, i) => (
                <p key={i} className="text-xs font-mono text-gray-500">{l}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
