import axios from "axios";

function resolveBaseUrl() {
  const configured = import.meta.env.VITE_API_URL;

  if (typeof window === "undefined") {
    return configured ?? "http://localhost:8000";
  }

  const { hostname, protocol } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return `${protocol}//localhost:8000`;
  }

  return configured ?? `${protocol}//${hostname}:8000`;
}

const BASE_URL = resolveBaseUrl();

export const api = axios.create({ baseURL: BASE_URL });

export const createSession = () => api.post("/sessions");

export const uploadAsset = (
  sessionId: string,
  assetType: string,
  file: File,
  view: "front" | "back" | "left" | "right" = "front",
) => {
  const form = new FormData();
  form.append("asset_type", assetType);
  form.append("file", file);
  form.append("view", view);
  return api.post(`/sessions/${sessionId}/assets`, form);
};

export const getStatus = (sessionId: string) =>
  api.get(`/sessions/${sessionId}/status`);

export const finalizeSession = (sessionId: string) =>
  api.post(`/sessions/${sessionId}/finalize`);

export const getReviewDashboard = () =>
  api.get("/admin/review");

export const registerSessionCharacter = (sessionId: string, characterNo: number) =>
  api.post(`/admin/session-queue/${sessionId}/character`, { character_no: characterNo });

export const cancelReviewSession = (queueName: "session-queue" | "unity-queue" | "history-queue", sessionId: string) =>
  api.delete(`/admin/review/${queueName}/${sessionId}`);

export const getHistorySessionInfo = (sessionId: string) =>
  api.get(`/admin/review/history-queue/${sessionId}/info`);

export const getAdminDashboard = () =>
  api.get("/admin/dashboard");

export const clearAdminQueue = (queueName: "lego-character" | "lego-object") =>
  api.delete(`/admin/queues/${queueName}`);

export const clearUnityQueue = () =>
  api.delete("/admin/unity-queue");

export const removeUnityQueueSession = (sessionId: string) =>
  api.delete(`/sessions/unity-queue/${sessionId}`);

export const cancelAdminSession = (sessionId: string) =>
  api.post(`/sessions/${sessionId}/cancel`);

export const deleteAdminSession = (sessionId: string) =>
  api.delete(`/admin/sessions/${sessionId}`);

export const resetAdminDatabase = () =>
  api.delete("/admin/db/reset");
