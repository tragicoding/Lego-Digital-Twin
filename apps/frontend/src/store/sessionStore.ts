import { create } from "zustand";

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
