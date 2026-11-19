import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The proxy forwards /api calls to FastAPI, so no CORS setup is needed
// in app.py during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
