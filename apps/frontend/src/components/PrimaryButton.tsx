import { ReactNode } from "react";
import { motion } from "framer-motion";

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
  if (variant === "secondary") {
    return (
      <motion.button
        onClick={onClick}
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
      onClick={onClick}
      disabled={disabled}
      whileTap={{ scale: 0.97 }}
      className="w-full py-4 rounded-2xl text-white font-bold text-base shadow-md disabled:opacity-40"
      style={{ background: "linear-gradient(135deg, #7C3AED, #38BDF8)" }}
    >
      {children}
    </motion.button>
  );
}
