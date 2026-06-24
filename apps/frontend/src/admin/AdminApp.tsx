import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  cancelAdminSession,
  clearAdminQueue,
  clearUnityQueue,
  deleteAdminSession,
  getAdminDashboard,
  resetAdminDatabase,
  updateCharacterComposition,
} from "../api/client";

type AssetSummary = {
  asset_id: string;
  asset_type: string;
  status: string;
  stage: string;
  progress: number;
};

type SessionSummary = {
  session_id: string;
  status: string;
  character_name: string | null;
  object_name: string | null;
  character_bottom: number | null;
  character_middle: number | null;
  character_top: number | null;
  character_number: number | null;
  character_selection_ready: boolean;
  likes: number;
  queue_position?: number | null;
  assets: AssetSummary[];
};

type Dashboard = {
  server: {
    backend_host: string;
    frontend_api_url: string | null;
    configured_windows_ip: string | null;
    ports: {
      localhost: Record<string, boolean>;
      configured_windows_ip: Record<string, boolean>;
    };
  };
  workers: {
    workers: Array<{ name: string; state: string; queues: string[]; current_job_id: string | null }>;
    queues: Array<{ name: string; count: number }>;
  };
  redis: {
    healthy: boolean;
    unity_queue_length: number;
  };
  data: {
    counts: Record<string, number>;
    sessions: SessionSummary[];
  };
  exhibition: {
    current_session_id: string | null;
    unity_queue: SessionSummary[];
  };
};

function statusLabel(ok: boolean) {
  return ok ? "OK" : "DOWN";
}

export default function AdminApp() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<SessionSummary | null>(null);
  const [composition, setComposition] = useState({ bottom: 1, middle: 1, top: 1 });

  const refresh = async () => {
    try {
      const res = await getAdminDashboard();
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

  const run = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await action();
      await refresh();
    } catch {
      setError(`${label} 처리에 실패했습니다.`);
    } finally {
      setBusy(null);
    }
  };

  const sessions = dashboard?.data.sessions ?? [];

  const openSession = (session: SessionSummary) => {
    setSelectedSession(session);
    setComposition({
      bottom: session.character_bottom ?? 1,
      middle: session.character_middle ?? 1,
      top: session.character_top ?? 1,
    });
  };

  const saveComposition = async () => {
    if (!selectedSession) return;
    await run("캐릭터 조합 저장", () =>
      updateCharacterComposition(selectedSession.session_id, composition),
    );
    setSelectedSession(null);
  };

  const calculatedCharacterNumber =
    (composition.bottom - 1) * 9 + (composition.middle - 1) * 3 + composition.top;

  return (
    <div className="min-h-screen bg-neutral-950 px-5 py-6 text-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
          <div>
            <h1 className="text-2xl font-black">MINIVERSE Admin</h1>
            <p className="mt-1 text-sm text-white/55">
              current: {dashboard?.exhibition.current_session_id ?? "-"} · 3초마다 자동 갱신
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

        <section className="grid gap-4 lg:grid-cols-3">
          <div className="rounded-md border border-white/10 bg-white/[0.03] p-4">
            <h2 className="mb-3 text-lg font-bold">Server</h2>
            <div className="space-y-2 text-sm text-white/70">
              <div>backend: {dashboard?.server.backend_host ?? "-"}</div>
              <div>windows ip: {dashboard?.server.configured_windows_ip ?? "-"}</div>
              <div>redis: {dashboard ? statusLabel(dashboard.redis.healthy) : "-"}</div>
            </div>
          </div>

          <div className="rounded-md border border-white/10 bg-white/[0.03] p-4">
            <h2 className="mb-3 text-lg font-bold">Workers</h2>
            <div className="space-y-2 text-sm text-white/70">
              {(dashboard?.workers.queues ?? []).map((queue) => (
                <div key={queue.name}>
                  {queue.name}: {queue.count} queued
                </div>
              ))}
              {(dashboard?.workers.workers ?? []).length === 0 && <div>실행 중인 worker 없음</div>}
            </div>
          </div>

          <div className="rounded-md border border-white/10 bg-white/[0.03] p-4">
            <h2 className="mb-3 text-lg font-bold">Data</h2>
            <div className="space-y-2 text-sm text-white/70">
              {Object.entries(dashboard?.data.counts ?? {}).map(([key, value]) => (
                <div key={key}>
                  {key}: {value}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-md border border-white/10 bg-white/[0.03] p-4">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-bold">Unity Queue</h2>
              <p className="text-xs text-white/45">{dashboard?.redis.unity_queue_length ?? 0} sessions waiting</p>
            </div>
            <button
              onClick={() => void run("Unity queue 비우기", clearUnityQueue)}
              disabled={busy !== null}
              className="rounded-md border border-white/20 px-3 py-2 text-sm font-semibold hover:bg-white/10 disabled:opacity-40"
            >
              비우기
            </button>
          </div>

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {(dashboard?.exhibition.unity_queue ?? []).map((session) => (
              <SessionCard key={session.session_id} session={session} onOpen={openSession} />
            ))}
            {(dashboard?.exhibition.unity_queue ?? []).length === 0 && (
              <div className="rounded-md border border-dashed border-white/10 p-6 text-center text-sm text-white/35">
                대기 중인 세션 없음
              </div>
            )}
          </div>
        </section>

        <section className="rounded-md border border-white/10 bg-white/[0.03] p-4">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-bold">Sessions</h2>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => void run("character queue 비우기", () => clearAdminQueue("lego-character"))}
                disabled={busy !== null}
                className="rounded-md border border-white/20 px-3 py-2 text-sm font-semibold hover:bg-white/10 disabled:opacity-40"
              >
                character queue 비우기
              </button>
              <button
                onClick={() => void run("object queue 비우기", () => clearAdminQueue("lego-object"))}
                disabled={busy !== null}
                className="rounded-md border border-white/20 px-3 py-2 text-sm font-semibold hover:bg-white/10 disabled:opacity-40"
              >
                object queue 비우기
              </button>
              <button
                onClick={() => void run("DB reset", resetAdminDatabase)}
                disabled={busy !== null}
                className="rounded-md border border-red-400/50 px-3 py-2 text-sm font-semibold text-red-200 hover:bg-red-500/10 disabled:opacity-40"
              >
                전체 reset
              </button>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {sessions.map((session) => (
              <SessionCard
                key={session.session_id}
                session={session}
                onOpen={openSession}
                actions={
                  <>
                    <button
                      onClick={() => void run("세션 취소", () => cancelAdminSession(session.session_id))}
                      disabled={busy !== null}
                      className="rounded-md border border-yellow-300/40 px-3 py-2 text-xs font-bold text-yellow-100 disabled:opacity-40"
                    >
                      취소
                    </button>
                    <button
                      onClick={() => void run("세션 삭제", () => deleteAdminSession(session.session_id))}
                      disabled={busy !== null}
                      className="rounded-md border border-red-400/50 px-3 py-2 text-xs font-bold text-red-200 disabled:opacity-40"
                    >
                      삭제
                    </button>
                  </>
                }
              />
            ))}
          </div>
        </section>
      </div>

      {selectedSession && (
        <CharacterCompositionModal
          session={selectedSession}
          composition={composition}
          characterNumber={calculatedCharacterNumber}
          busy={busy !== null}
          onChange={setComposition}
          onClose={() => setSelectedSession(null)}
          onSave={() => void saveComposition()}
        />
      )}
    </div>
  );
}

function SessionCard({
  session,
  actions,
  onOpen,
}: {
  session: SessionSummary;
  actions?: ReactNode;
  onOpen?: (session: SessionSummary) => void;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onOpen?.(session)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onOpen?.(session);
      }}
      className="rounded-md border border-white/10 bg-black/25 p-4 text-left hover:border-white/25"
    >
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <div className="font-bold">#{session.session_id}</div>
          <div className="text-xs text-white/45">{session.status}</div>
        </div>
        {session.queue_position && <div className="text-xs text-white/45">pos {session.queue_position}</div>}
      </div>
      <div className="space-y-1 text-xs text-white/65">
        <div>character: {session.character_name ?? "-"}</div>
        <div>object: {session.object_name ?? "-"}</div>
        <div>
          parts: bottom_{session.character_bottom ?? "-"} / middle_{session.character_middle ?? "-"} / top_
          {session.character_top ?? "-"}
        </div>
        <div>character no: {session.character_number ?? "미선택"}</div>
        <div>likes: {session.likes}</div>
      </div>
      <div className="mt-3 space-y-1 text-xs text-white/55">
        {session.assets.map((asset) => (
          <div key={asset.asset_id}>
            {asset.asset_type}: {asset.status} / {asset.progress}%
          </div>
        ))}
      </div>
      {actions && (
        <div
          className="mt-4 flex gap-2"
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
        >
          {actions}
        </div>
      )}
    </div>
  );
}

function CharacterCompositionModal({
  session,
  composition,
  characterNumber,
  busy,
  onChange,
  onClose,
  onSave,
}: {
  session: SessionSummary;
  composition: { bottom: number; middle: number; top: number };
  characterNumber: number;
  busy: boolean;
  onChange: (next: { bottom: number; middle: number; top: number }) => void;
  onClose: () => void;
  onSave: () => void;
}) {
  const updatePart = (key: "bottom" | "middle" | "top", value: string) => {
    const parsed = Number(value);
    onChange({ ...composition, [key]: parsed });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
      <div className="w-full max-w-md rounded-md border border-white/15 bg-neutral-950 p-5 text-white shadow-2xl">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-black">Session #{session.session_id}</h2>
            <p className="mt-1 text-sm text-white/50">캐릭터 조합을 선택하세요</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md border border-white/20 px-3 py-1 text-sm hover:bg-white/10"
          >
            닫기
          </button>
        </div>

        <div className="grid gap-3">
          <PartSelect
            label="하체 bottom"
            value={composition.bottom}
            onChange={(value) => updatePart("bottom", value)}
          />
          <PartSelect
            label="상체 middle"
            value={composition.middle}
            onChange={(value) => updatePart("middle", value)}
          />
          <PartSelect
            label="머리 top"
            value={composition.top}
            onChange={(value) => updatePart("top", value)}
          />
        </div>

        <div className="mt-5 rounded-md bg-white/[0.06] p-4 text-sm">
          <div className="text-white/55">Unity character number</div>
          <div className="mt-1 text-4xl font-black">{characterNumber}</div>
          <div className="mt-2 text-white/55">
            bottom_{composition.bottom} + middle_{composition.middle} + top_{composition.top}
          </div>
        </div>

        <button
          onClick={onSave}
          disabled={busy}
          className="mt-5 w-full rounded-md bg-white px-4 py-3 text-sm font-black text-black disabled:opacity-40"
        >
          저장하고 Unity Queue 조건 반영
        </button>
      </div>
    </div>
  );
}

function PartSelect({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: string) => void;
}) {
  return (
    <label className="grid gap-2 text-sm font-semibold text-white/75">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-white/15 bg-black px-3 py-3 text-base text-white"
      >
        <option value={1}>1</option>
        <option value={2}>2</option>
        <option value={3}>3</option>
      </select>
    </label>
  );
}
