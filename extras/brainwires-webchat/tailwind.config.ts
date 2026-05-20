import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bw: {
          bg: "#0a0a0f",
          surface: "#14141b",
          border: "#23232e",
          accent: "#e94560",
          user: "#2c3e50",
          assistant: "#16213e",
        },
      },
    },
  },
  plugins: [],
};

export default config;
