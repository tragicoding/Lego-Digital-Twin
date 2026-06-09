import { startTransition, useEffect, useState, type ReactNode } from "react";
import {
  cancelAdminSession,
  clearAdminQueue,
  clearUnityQueue,
  deleteAdminSession,
  getAdminDashboard,
  removeUnityQueueSession,
  resetAdminDatabase,
} from "../api/client";

type Dashboard = {
  server: {
    health: string;
    backend_host: string;
    frontend_api_url: string | null;
    configured_windows_ip: string | null;
    wsl_ips: string[];
    ports: {
      localhost: Record<string, boolean>;
      configured_windows_ip: Record<string, boolean>;
    };
  };
  workers: {
    workers: Array<{
      name: string;
      state: string;
      queues: string[];
      current_job_id: string | null;
      last_heartbeat: string | null;
    }>;
    queues: Array<{
      name: string;
      count: number;
      registries: Record<string, Array<{
        job_id: string;
        status: string;
        func_name: string;
        args: string[];
        created_at: string | null;
        description: string;
      }>>;
    }>;
  };
  redis: {
    healthy: boolean;
    key_counts: Record<string, number>;
    unity_queue_length: number;
  };
  data: {
    counts: Record<string, number>;
    models: Array<{
      name: string;
      size_bytes: number;
      updated_at: string;
    }>;
    sessions: SessionSummary[];
  };
  exhibition: {
    current_session_id: string | null;
    unity_queue: SessionSummary[];
    likes_ranking: Array<{
      session_id: string;
      character_name: string | null;
      object_name: string | null;
      likes: number;
    }>;
  };
};

type SessionSummary = {
  session_id: string;
  status: string;
  created_at: string | null;
  updated_at: string | null;
  character_name: string | null;
  object_name: string | null;
  signature_motion: string | null;
  likes: number;
  queue_position: number | null;
  image_checks: {
    character: ViewCheck;
    object: ViewCheck;
  };
  downloads: {
    character_fbx: boolean;
    character_texture_glb: boolean;
    object_glb: boolean;
  };
  assets: Array<{
    asset_id: string;
    asset_type: string;
    status: string;
    stage: string;
    progress: number;
    model_url: string | null;
    thumbnail_url: string | null;
    created_at: string | null;
  }>;
};

type ViewCheck = {
  front: boolean;
  left: boolean;
  back: boolean;
  right: boolean;
  all_present: boolean;
};

function Section({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-white/20 bg-white/5 p-5">
      <h2 className="text-lg font-bold text-white">{title}</h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function AdminButton({
  children,
  onClick,
  danger = false,
}: {
  children: ReactNode;
  onClick: () => void | Promise<void>;
  danger?: boolean;
}) {
  return (
    <button
      onClick={() => void onClick()}
      className={`rounded-xl border px-4 py-2 text-sm font-semibold transition ${
        danger
          ? "border-red-400/60 text-red-200 hover:bg-red-500/10"
          : "border-white/25 text-white hover:bg-white/10"
      }`}
    >
      {children}
    </button>
  );
}

function BoolPill({ value }: { value: boolean }) {
  return (
    <span
      className={`inline-flex rounded-full border px-2 py-1 text-xs font-semibold ${
        value
          ? "border-emerald-400/40 text-emerald-300"
          : "border-red-400/40 text-red-300"
      }`}
    >
      {value ? "true" : "false"}
    </span>
  );
}

function formatDate(value: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("ko-KR");
}

function SessionCard({
  session,
  onCancel,
  onDelete,
}: {
  session: SessionSummary;
  onCancel: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}) {
  return (
    <div className="rounded-xl border border-white/15 bg-black/20 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-bold text-white">
            #{session.session_id}
            {session.queue_position ? ` · queue ${session.queue_position}` : ""}
          </div>
          <div className="mt-1 text-xs text-white/60">
            {formatDate(session.created_at)}
          </div>
        </div>
        <div className="flex gap-2">
          <AdminButton danger onClick={() => onCancel(session.session_id)}>
            취소
          </AdminButton>
          <AdminButton danger onClick={() => onDelete(session.session_id)}>
            완전 삭제
          </AdminButton>
        </div>
      </div>

      <div className="mt-3 grid gap-2 text-sm text-white/85 md:grid-cols-2">
        <div>캐릭터 이름: {session.character_name ?? "-"}</div>
        <div>오브제 이름: {session.object_name ?? "-"}</div>
        <div>좋아요: {session.likes}</div>
        <div>시그니처 동작: {session.signature_motion ?? "-"}</div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-white/10 p-3">
          <div className="text-xs font-bold text-white/70">이미지 검사</div>
          <div className="mt-2 space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-white/80">character 4 views</span>
              <BoolPill value={session.image_checks.character.all_present} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-white/80">object 4 views</span>
              <BoolPill value={session.image_checks.object.all_present} />
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-white/10 p-3">
          <div className="text-xs font-bold text-white/70">다운로드 검사</div>
          <div className="mt-2 space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-white/80">character fbx</span>
              <BoolPill value={session.downloads.character_fbx} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-white/80">character glb</span>
              <BoolPill value={session.downloads.character_texture_glb} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-white/80">object glb</span>
              <BoolPill value={session.downloads.object_glb} />
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full text-left text-xs text-white/75">
          <thead className="text-white/50">
            <tr>
              <th className="pb-2 pr-4">asset</th>
              <th className="pb-2 pr-4">status</th>
              <th className="pb-2 pr-4">stage</th>
              <th className="pb-2 pr-4">progress</th>
            </tr>
          </thead>
          <tbody>
            {session.assets.map((asset) => (
              <tr key={asset.asset_id} className="border-t border-white/10">
                <td className="py-2 pr-4">{asset.asset_type}</td>
                <td className="py-2 pr-4">{asset.status}</td>
                <td className="py-2 pr-4">{asset.stage}</td>
                <td className="py-2 pr-4">{asset.progress}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function AdminApp() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    try {
      const res = await getAdminDashboard();
      startTransition(() => {
        setDashboard(res.data);
        setError(null);
        setLoading(false);
      });
    } catch (err) {
      console.error(err);
      setError("관리자 데이터를 불러오지 못했습니다.");
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  const runAction = async (action: () => Promise<unknown>, confirmText?: string) => {
    if (confirmText && !window.confirm(confirmText)) return;
    await action();
    await refresh();
  };

  if (loading && !dashboard) {
    return (
      <div className="min-h-screen bg-black px-6 py-8 text-white">
        관리자 데이터 로딩 중...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black px-6 py-8 text-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/20 bg-white/5 p-5">
          <div>
            <h1 className="text-2xl font-black">MINIVERSE Admin</h1>
            <p className="mt-1 text-sm text-white/60">
              새로고침 없이 5초마다 자동 업데이트됩니다.
            </p>
            {error ? <p className="mt-2 text-sm text-red-300">{error}</p> : null}
          </div>
          <div className="flex flex-wrap gap-2">
            <AdminButton onClick={refresh}>새로고침</AdminButton>
            <AdminButton
              danger
              onClick={() => runAction(resetAdminDatabase, "DB와 큐를 모두 초기화할까요?")}
            >
              DB 초기화
            </AdminButton>
          </div>
        </header>

        <Section title="서버">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">health</div>
              <div className="mt-2 text-lg font-bold">{dashboard?.server.health}</div>
            </div>
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">configured Windows IP</div>
              <div className="mt-2 text-lg font-bold">
                {dashboard?.server.configured_windows_ip ?? "-"}
              </div>
            </div>
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">WSL IP</div>
              <div className="mt-2 text-sm font-semibold">
                {dashboard?.server.wsl_ips.join(", ") || "-"}
              </div>
            </div>
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">backend host</div>
              <div className="mt-2 break-all text-sm font-semibold">
                {dashboard?.server.backend_host}
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {Object.entries(dashboard?.server.ports.localhost ?? {}).map(([name, value]) => (
              <div
                key={name}
                className="flex items-center justify-between rounded-xl border border-white/10 p-4"
              >
                <span className="text-sm text-white/80">localhost {name}</span>
                <BoolPill value={value} />
              </div>
            ))}
            {Object.entries(dashboard?.server.ports.configured_windows_ip ?? {}).map(([name, value]) => (
              <div
                key={name}
                className="flex items-center justify-between rounded-xl border border-white/10 p-4"
              >
                <span className="text-sm text-white/80">windows {name}</span>
                <BoolPill value={value} />
              </div>
            ))}
          </div>
        </Section>

        <Section title="워커 / 큐">
          <div className="grid gap-4 xl:grid-cols-2">
            <div className="space-y-4">
              <div className="rounded-xl border border-white/10 p-4">
                <div className="text-sm font-bold">Worker 상태</div>
                <div className="mt-3 space-y-3">
                  {dashboard?.workers.workers.map((worker) => (
                    <div
                      key={worker.name}
                      className="rounded-xl border border-white/10 p-3 text-sm"
                    >
                      <div className="font-semibold text-white">{worker.name}</div>
                      <div className="mt-1 text-white/70">state: {worker.state}</div>
                      <div className="text-white/70">
                        queues: {worker.queues.join(", ")}
                      </div>
                      <div className="text-white/70">
                        heartbeat: {formatDate(worker.last_heartbeat)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {dashboard?.workers.queues.map((queue) => (
                <div key={queue.name} className="rounded-xl border border-white/10 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-bold">{queue.name}</div>
                      <div className="text-xs text-white/60">queued count: {queue.count}</div>
                    </div>
                    <AdminButton
                      danger
                      onClick={() =>
                        runAction(
                          () => clearAdminQueue(queue.name as "lego-character" | "lego-object"),
                          `${queue.name} 큐를 비울까요?`,
                        )
                      }
                    >
                      queue 비우기
                    </AdminButton>
                  </div>

                  <div className="mt-3 space-y-3">
                    {Object.entries(queue.registries).map(([registryName, jobs]) => (
                      <div key={registryName} className="rounded-xl border border-white/10 p-3">
                        <div className="text-xs font-bold text-white/60">
                          {registryName} ({jobs.length})
                        </div>
                        <div className="mt-2 max-h-40 space-y-2 overflow-auto text-xs text-white/70">
                          {jobs.length === 0 ? (
                            <div>비어 있음</div>
                          ) : (
                            jobs.map((job) => (
                              <div key={job.job_id} className="rounded-lg border border-white/10 p-2">
                                <div className="font-semibold text-white">{job.func_name}</div>
                                <div>{job.job_id}</div>
                                <div>{job.description}</div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Section>

        <Section title="Redis / 전시">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">Redis health</div>
              <div className="mt-2">
                <BoolPill value={Boolean(dashboard?.redis.healthy)} />
              </div>
            </div>
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">현재 진행 세션</div>
              <div className="mt-2 text-lg font-bold">
                {dashboard?.exhibition.current_session_id ?? "-"}
              </div>
            </div>
            <div className="rounded-xl border border-white/10 p-4">
              <div className="text-xs text-white/50">unity_queue 길이</div>
              <div className="mt-2 text-lg font-bold">
                {dashboard?.redis.unity_queue_length ?? 0}
              </div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <AdminButton
              danger
              onClick={() => runAction(clearUnityQueue, "unity_queue를 비울까요?")}
            >
              unity_queue 비우기
            </AdminButton>
          </div>

          <div className="mt-4 space-y-4">
            {dashboard?.exhibition.unity_queue.map((session) => (
              <div key={session.session_id} className="rounded-xl border border-white/10 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <div className="font-bold text-white">
                      queue #{session.queue_position} · {session.session_id}
                    </div>
                    <div className="text-xs text-white/60">
                      {session.character_name ?? "-"} / {session.object_name ?? "-"}
                    </div>
                  </div>
                  <AdminButton
                    danger
                    onClick={() =>
                      runAction(
                        () => removeUnityQueueSession(session.session_id),
                        `${session.session_id}를 unity_queue에서 제거할까요?`,
                      )
                    }
                  >
                    queue에서 제거
                  </AdminButton>
                </div>
                <SessionCard
                  session={session}
                  onCancel={(sessionId) =>
                    runAction(
                      () => cancelAdminSession(sessionId),
                      `${sessionId} 세션을 취소할까요? 이후 업로드와 worker 진행이 중단됩니다.`,
                    )
                  }
                  onDelete={(sessionId) =>
                    runAction(
                      () => deleteAdminSession(sessionId),
                      `${sessionId} 세션 자체를 삭제할까요?`,
                    )
                  }
                />
              </div>
            ))}
          </div>
        </Section>

        <Section title="데이터 관리">
          <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
            <div className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {Object.entries(dashboard?.data.counts ?? {}).map(([key, value]) => (
                  <div key={key} className="rounded-xl border border-white/10 p-4">
                    <div className="text-xs text-white/50">{key}</div>
                    <div className="mt-2 text-xl font-bold">{value}</div>
                  </div>
                ))}
              </div>

              <div className="rounded-xl border border-white/10 p-4">
                <div className="text-sm font-bold">세션 목록</div>
                <div className="mt-4 space-y-4">
                  {dashboard?.data.sessions.map((session) => (
                    <SessionCard
                      key={session.session_id}
                      session={session}
                      onCancel={(sessionId) =>
                        runAction(
                          () => cancelAdminSession(sessionId),
                          `${sessionId} 세션을 취소할까요? 이후 업로드와 worker 진행이 중단됩니다.`,
                        )
                      }
                      onDelete={(sessionId) =>
                        runAction(
                          () => deleteAdminSession(sessionId),
                          `${sessionId} 세션을 삭제할까요?`,
                        )
                      }
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-xl border border-white/10 p-4">
                <div className="text-sm font-bold">생성된 모델 목록</div>
                <div className="mt-3 max-h-[28rem] space-y-2 overflow-auto text-sm text-white/75">
                  {dashboard?.data.models.map((model) => (
                    <div key={model.name} className="rounded-lg border border-white/10 p-3">
                      <div className="font-semibold text-white">{model.name}</div>
                      <div>{(model.size_bytes / 1024).toFixed(1)} KB</div>
                      <div>{formatDate(model.updated_at)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-white/10 p-4">
                <div className="text-sm font-bold">좋아요 순위</div>
                <div className="mt-3 space-y-2 text-sm text-white/75">
                  {dashboard?.exhibition.likes_ranking.map((item, index) => (
                    <div
                      key={item.session_id}
                      className="flex items-center justify-between rounded-lg border border-white/10 p-3"
                    >
                      <span>
                        {index + 1}. {item.session_id} {item.character_name ?? "-"}
                      </span>
                      <span className="font-semibold text-white">♥ {item.likes}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
