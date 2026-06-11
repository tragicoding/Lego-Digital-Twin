/*
React 프론트엔드가 FastAPI 백엔드와 통신하기 위해 Axios로 만든 API 함수 모음 파일
StartPage
→ createSession()

CapturePage
→ uploadAsset()

ProfilePage
→ updateProfile()

LoadingPage
→ getStatus()
*/


//axios는 프론트엔드에서 백엔드 API를 호출할 때 쓰는 HTTP 요청 라이브러리
import axios from "axios";

function resolveBaseUrl() {
  const configured = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

  if (typeof window === "undefined") {
    return configured;
  }

  const { hostname, protocol } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return `${protocol}//localhost:8000`;
  }

  return configured;
}

//백엔드 서버 주소 정한다.
//로컬 PC에서 localhost로 admin/frontend를 열면 localhost:8000을 우선 사용하고,
//태블릿이나 다른 기기에서 접속할 때는 .env의 VITE_API_URL을 사용한다.
const BASE_URL = resolveBaseUrl();

//Axios 인스턴스 생성.
//api라는 이름으로 기능들에 접근. 
export const api = axios.create({ baseURL: BASE_URL });

//새로운 세션을 생성하는 API함수.
//StartPage에서 사용한다.
//const res = await createSession();
//세션 생성 요청을 백엔드에 보내고, 응답으로 session_id를 받는다.
export const createSession = () => api.post("/sessions");

//이미지 파일을 백엔드에 업로드하는 함수.
//인자 : 
/*
sessionId  → 현재 촬영 세션 ID
assetType  → character / building / vehicle
file       → 사용자가 선택한 이미지 파일

FormData는 파일 업로드할 때 사용하는 브라우저 내장 객체
백엔드로 보낼 데이터를 form에 담아서 보낸다.

실제 요청 주소:
POST /sessions/{sessionID}/assets
예를들어,
POST /sessions/abc123/assets
form 안에는
asset_type: "character"
file: 이미지 파일
*/
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



/*
사용자 프로필 정보를 백엔드에 업데이트하는 함수
PATCH는 기존 데이터 중 일부만 수정할 때 많이 쓴다.
nickname은 필수고, 나머지는 선택값.
*/
export const updateProfile = (sessionId: string, data: {
  nickname?: string;
  character_npc_name?: string;
  object_name?: string;
  phone?: string;
  bubble_text?: string;
  favorite_theme?: string;
}) => api.patch(`/sessions/${sessionId}/profile`, data);

/*
세션의 현재 상태를 백엔드에서 조회하는 함수
LoadingPage에서 사용한다.
const res = await getStatus(sessionID);
세션 ID를 백엔드에 보내면, 현재 각 모델의 변환 상태와 Unity로 이동할 준비가 되었는지 여부를 응답으로 받는다.
GET /sessions/{sessionId}/status
→ character/building/vehicle 변환 상태 확인
→ ready_for_unity 여부 확인
*/
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
