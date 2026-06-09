import { create } from "zustand";

interface SessionStore {
  sessionId: string | null;
  setSessionId: (id: string) => void;

  characterAssetId: string | null;
  setCharacterAssetId: (id: string) => void;

  objectAssetId: string | null;
  setObjectAssetId: (id: string) => void;

  characterNpcName: string;
  setCharacterNpcName: (name: string) => void;

  objectName: string;
  setObjectName: (name: string) => void;

  // 레거시 호환 (CapturePage 진행 상태)
  uploads: Record<string, boolean>;
  setUploaded: (type: string) => void;

  reset: () => void;
}

const INITIAL_STATE = {
  sessionId: null,
  characterAssetId: null,
  objectAssetId: null,
  characterNpcName: "",
  objectName: "",
  uploads: {},
};

export const useSessionStore = create<SessionStore>((set) => ({
  ...INITIAL_STATE,

  setSessionId: (id) => set({ sessionId: id }),
  setCharacterAssetId: (id) => set({ characterAssetId: id }),
  setObjectAssetId: (id) => set({ objectAssetId: id }),
  setCharacterNpcName: (name) => set({ characterNpcName: name }),
  setObjectName: (name) => set({ objectName: name }),
  setUploaded: (type) =>
    set((s) => ({ uploads: { ...s.uploads, [type]: true } })),

  reset: () => set(INITIAL_STATE),
}));
