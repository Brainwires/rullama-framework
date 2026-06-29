import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { Topbar } from "@/components/layout/topbar";
import "./globals.css";


export const metadata: Metadata = {
  metadataBase: new URL("https://brainwires.dev"),
  title: { default: "Brainwires — Rust AI Agent Framework", template: "%s | Brainwires" },
  description: "Documentation for the Brainwires Agent Framework — a modular, production-grade Rust framework for building AI agents.",
  openGraph: { siteName: "brainwires.dev", type: "website" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <SidebarProvider>
            <AppSidebar />
            <div className="flex flex-1 flex-col min-w-0">
              <Topbar />
              <main className="flex-1">{children}</main>
            </div>
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
