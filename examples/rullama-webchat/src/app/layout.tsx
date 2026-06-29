import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BrainClaw Chat",
  description: "Browser chat client for the BrainClaw personal AI assistant.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
