import { useEffect, useState } from "react";
import {
  cancelReviewSession,
  getHistorySessionInfo,
  getReviewDashboard,
  registerSessionCharacter,
} from "../api/client";

type QueueName = "session_queue" | "unity_queue" | "history_queue";
type ApiQueueName = "session-queue" | "unity-queue" | "history-queue";

type QueueItem = {
  session_id: string;
  position: number;
  status: string;
  created_at: string | null;
  updated_at: string | null;
  character_no: string | null;
  object_status: string;
  object_stage: string;
  object_progress: number;
  object_model_url: string | null;
  object_error: string | null;
};

type Dashboard = {
  active_session_id: string | null;
  session_queue: QueueItem[];
  unity_queue: QueueItem[];
  history_queue: QueueItem[];
};

type ModalState = {
  queue: QueueName;
  item: QueueItem;
} | null;

const QUEUES: Array<{ key: QueueName; api: ApiQueueName; title: string; hint: string }> = [
  { key: "session_queue", api: "session-queue", title: "session_queue", hint: "앱에서 진행 중인 세션" },
  { key: "unity_queue", api: "unity-queue", title: "unity_queue", hint: "Unity 입장 대기 세션" },
  { key: "history_queue", api: "history-queue", title: "history_queue", hint: "Unity 종료 후 검수 세션" },
];

function fmt(value: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("ko-KR");
}

function apiName(queue: QueueName): ApiQueueName {
  return queue.replace("_", "-") as ApiQueueName;
}

export default function AdminApp() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [selected, setSelected] = useState<ModalState>(null);
  const [characterNo, setCharacterNo] = useState("");
  const [info, setInfo] = useState<Record<string, string | null> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const res = await getReviewDashboard();
      setDashboard(res.data);
      setError(null);
    } catch {
      setError("관리자 데이터를 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, []);

  const open = (queue: QueueName, item: QueueItem) => {
    setSelected({ queue, item });
    setCharacterNo(item.character_no ?? "");
    setInfo(null);
  };

  const close = () => {
    setSelected(null);
    setInfo(null);
  };

  const register = async () => {
    if (!selected || !characterNo.trim()) return;
    await registerSessionCharacter(selected.item.session_id, Number(characterNo));
    await refresh();
    close();
  };

  const cancel = async () => {
    if (!selected) return;
    await cancelReviewSession(apiName(selected.queue), selected.item.session_id);
    await refresh();
    close();
  };

  const showInfo = async () => {
    if (!selected) return;
    const res = await getHistorySessionInfo(selected.item.session_id);
    setInfo(res.data);
  };

  return (
    <div className="min-h-screen bg-neutral-950 px-5 py-6 text-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
          <div>
            <h1 className="text-2xl font-black">MINIVERSE Admin</h1>
            <p className="mt-1 text-sm text-white/55">
              active: {dashboard?.active_session_id ?? "-"} · 3초마다 자동 갱신
            </p>
          </div>
          <button
            onClick={() => void refresh()}
            className="rounded-md border border-white/20 px-4 py-2 text-sm font-semibold hover:bg-white/10"
          >
            새로고침
          </button>
        </header>

        {error && <p className="rounded-md bg-red-500/15 px-4 py-3 text-sm text-red-200">{error}</p>}

        <main className="grid gap-4 lg:grid-cols-3">
          {QUEUES.map((queue) => (
            <section key={queue.key} className="min-h-[28rem] border border-white/10 bg-white/[0.03] p-4">
              <div className="mb-4">
                <h2 className="text-lg font-bold">{queue.title}</h2>
                <p className="text-xs text-white/45">{queue.hint}</p>
              </div>

              <div className="space-y-3">
                {(dashboard?.[queue.key] ?? []).map((item) => (
                  <button
                    key={item.session_id}
                    onClick={() => open(queue.key, item)}
                    className="w-full rounded-md border border-white/10 bg-black/25 p-4 text-left hover:border-white/35"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-bold">#{item.session_id}</span>
                      <span className="text-xs text-white/45">pos {item.position}</span>
                    </div>
                    <div className="mt-2 grid gap-1 text-xs text-white/65">
                      <span>character: {item.character_no ?? "-"}</span>
                      <span>object: {item.object_status} / {item.object_progress}%</span>
                      <span>updated: {fmt(item.updated_at)}</span>
                    </div>
                  </button>
                ))}

                {(dashboard?.[queue.key] ?? []).length === 0 && (
                  <div className="rounded-md border border-dashed border-white/10 p-6 text-center text-sm text-white/35">
                    비어 있음
                  </div>
                )}
              </div>
            </section>
          ))}
        </main>
      </div>

      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-md rounded-md border border-white/15 bg-neutral-950 p-5 shadow-2xl">
            <div className="mb-5">
              <div className="text-xs text-white/45">{selected.queue}</div>
              <h3 className="text-xl font-black">#{selected.item.session_id}</h3>
            </div>

            <div className="space-y-3 text-sm text-white/75">
              <div>캐릭터 번호: {selected.item.character_no ?? "-"}</div>
              <div>오브제 상태: {selected.item.object_status} / {selected.item.object_stage}</div>
              {selected.item.object_error && <div className="text-red-300">{selected.item.object_error}</div>}
            </div>

            {selected.queue === "session_queue" && (
              <div className="mt-5 space-y-3">
                <input
                  value={characterNo}
                  onChange={(e) => setCharacterNo(e.target.value)}
                  placeholder="캐릭터 번호 입력 (1-5)"
                  className="w-full rounded-md border border-white/15 bg-black px-3 py-3 text-white outline-none focus:border-white/40"
                  inputMode="numeric"
                />
                <button onClick={register} className="w-full rounded-md bg-white px-4 py-3 font-bold text-black">
                  등록하기
                </button>
              </div>
            )}

            {selected.queue === "history_queue" && (
              <div className="mt-5 space-y-3">
                <button onClick={showInfo} className="w-full rounded-md border border-white/20 px-4 py-3 font-bold">
                  정보
                </button>
                {info && (
                  <div className="rounded-md bg-white/5 p-3 text-sm text-white/75">
                    <div>캐릭터 이름: {info.character_name ?? "-"}</div>
                    <div>오브제 이름: {info.object_name ?? "-"}</div>
                    <div>bubble_text: {info.bubble_text ?? "-"}</div>
                  </div>
                )}
              </div>
            )}

            <div className="mt-5 grid grid-cols-2 gap-3">
              <button onClick={close} className="rounded-md border border-white/20 px-4 py-3 font-bold">
                뒤로
              </button>
              <button onClick={cancel} className="rounded-md border border-red-400/50 px-4 py-3 font-bold text-red-200">
                취소하기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
