import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL: BASE_URL });

export const createSession = () => api.post("/sessions");

export const uploadAsset = (sessionId: string, assetType: string, file: File) => {
  const form = new FormData();
  form.append("asset_type", assetType);
  form.append("file", file);
  return api.post(`/sessions/${sessionId}/assets`, form);
};

export const updateProfile = (sessionId: string, data: {
  nickname: string;
  phone?: string;
  bubble_text?: string;
  favorite_theme?: string;
}) => api.patch(`/sessions/${sessionId}/profile`, data);

export const getStatus = (sessionId: string) =>
  api.get(`/sessions/${sessionId}/status`);
