import type { ReactNode } from "react";

/**
 * 수평 중앙 정렬 + 최대 너비 제한만 담당.
 * 세로 높이/중앙 정렬은 각 페이지가 min-h-screen으로 직접 처리.
 */
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <div className="w-full bg-white">
      <div className="w-full max-w-md mx-auto">
        {children}
      </div>
    </div>
  );
}
