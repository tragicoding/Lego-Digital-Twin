import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "admin",
  publicDir: "../public",
  plugins: [react()],
  server: {
    port: 3001,
    strictPort: true,
    host: true,
  },
  build: {
    outDir: "../dist-admin",
    emptyOutDir: true,
  },
});
