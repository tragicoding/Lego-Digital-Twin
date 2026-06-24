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

export const prepareCharacterPlaceholder = (sessionId: string) =>
  api.post(`/sessions/${sessionId}/character-placeholder`);

export const captureObjectFromCameras = (sessionId: string) =>
  api.post(`/sessions/${sessionId}/capture/object`);

export const getCameraHealth = () =>
  api.get("/camera/health");

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

export const updateCharacterComposition = (
  sessionId: string,
  body: { bottom: number; middle: number; top: number },
) => api.patch(`/admin/sessions/${sessionId}/character-composition`, body);
