import type { ReactNode } from "react";
import { motion } from "framer-motion";

// 효과음 파일 위치: apps/frontend/public/sounds/button-click.mp3
const clickSound = new Audio("/sounds/button-click.mp3");
clickSound.volume = 0.4;

function playClickSound() {
  clickSound.currentTime = 0;
  clickSound.play().catch(() => {});
}

interface Props {
  onClick?: () => void;
  disabled?: boolean;
  children: ReactNode;
  variant?: "primary" | "secondary";
}

export default function PrimaryButton({
  onClick,
  disabled,
  children,
  variant = "primary",
}: Props) {
  function handleClick() {
    playClickSound();
    onClick?.();
  }

  if (variant === "secondary") {
    return (
      <motion.button
        onClick={handleClick}
        disabled={disabled}
        whileTap={{ scale: 0.97 }}
        className="w-full py-4 rounded-2xl border-2 border-gray-200 text-gray-600 font-semibold text-base bg-white disabled:opacity-40"
      >
        {children}
      </motion.button>
    );
  }

  return (
    <motion.button
      onClick={handleClick}
      disabled={disabled}
      whileTap={{ scale: 0.97 }}
      className="w-full py-4 rounded-2xl text-white font-bold text-base shadow-md disabled:opacity-40"
      style={{ background: "linear-gradient(135deg, #7C3AED, #38BDF8)" }}
    >
      {children}
    </motion.button>
  );
}
