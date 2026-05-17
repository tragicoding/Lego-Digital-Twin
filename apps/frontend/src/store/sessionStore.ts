/*전역 상태
session_id 하나를 전역에서 공유.
 모든 페이지가 이걸 참조한다.
typeScript + Zustand 전역 상태 저장소 코드
session_id는 서버에서 생성되고, 클라이언트에서 참조한다.
 session_id는 string이거나 null일 수 있다. 
 초기값은 null이다.
세션 ID와 업로드 완료 여부를 여러 페이지에서 공유하기 
위한 파일
SessionID, setSessionID함수,uploads객체, setUploaded함수


세션 ID와 캐릭터/건축물/자동차 업로드 완료 여부를 모든 페이지에서 
공유하기 위해 만든 Zustand 전역 상태 관리 코드

*/
import { create } from "zustand";

//Type Script
//SessionStore 라는 전역 상태의 구조 미리 정의.
interface SessionStore {
  sessionId: string | null;
  setSessionId: (id: string) => void;
  uploads: Record<string, boolean>; // character/building/vehicle → uploaded?
  setUploaded: (type: string) => void;
}

export const useSessionStore = create<SessionStore>((set) => ({
  sessionId: null,
  setSessionId: (id) => set({ sessionId: id }),
  uploads: {},
  setUploaded: (type) =>
    set((s) => ({ uploads: { ...s.uploads, [type]: true } })),
}));
