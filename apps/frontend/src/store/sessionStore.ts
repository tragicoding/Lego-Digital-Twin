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
}

export const useSessionStore = create<SessionStore>((set) => ({
  sessionId: null,
  setSessionId: (id) => set({ sessionId: id }),

  characterAssetId: null,
  setCharacterAssetId: (id) => set({ characterAssetId: id }),

  objectAssetId: null,
  setObjectAssetId: (id) => set({ objectAssetId: id }),

  characterNpcName: "",
  setCharacterNpcName: (name) => set({ characterNpcName: name }),

  objectName: "",
  setObjectName: (name) => set({ objectName: name }),

  uploads: {},
  setUploaded: (type) =>
    set((s) => ({ uploads: { ...s.uploads, [type]: true } })),
}));
