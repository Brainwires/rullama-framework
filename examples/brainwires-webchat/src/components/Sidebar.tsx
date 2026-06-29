"use client";

import { useRouter } from "next/navigation";

export interface SidebarProps {
  sessionId: string | null;
  connection: string;
}

export function Sidebar({ sessionId, connection }: SidebarProps) {
  const router = useRouter();

  async function logout() {
    await fetch("/api/auth", { method: "DELETE" });
    router.replace("/login");
  }

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-bw-border bg-bw-surface md:flex">
      <div className="border-b border-bw-border px-4 py-3">
        <h1 className="text-base font-semibold text-bw-accent">BrainClaw</h1>
        <p className="text-xs text-neutral-400">Personal AI Chat</p>
      </div>
      <div className="flex-1 overflow-auto scrollbar-thin px-4 py-3">
        <div>
          <h2 className="mb-2 text-[11px] uppercase tracking-wide text-neutral-500">
            Current session
          </h2>
          <div className="truncate rounded bg-bw-bg px-2 py-1 text-xs text-neutral-200">
            {sessionId ?? "…connecting"}
          </div>
        </div>
        <div className="mt-6">
          <h2 className="mb-2 text-[11px] uppercase tracking-wide text-neutral-500">
            Connection
          </h2>
          <div className="text-xs text-neutral-200">{connection}</div>
        </div>
        <div className="mt-6">
          <h2 className="mb-2 text-[11px] uppercase tracking-wide text-neutral-500">
            Tips
          </h2>
          <ul className="space-y-1 text-xs text-neutral-400">
            <li>
              Type <code className="text-neutral-200">/status</code> to inspect
              the agent.
            </li>
            <li>
              Type <code className="text-neutral-200">/think high</code> to
              enable extended thinking.
            </li>
            <li>
              <code className="text-neutral-200">/new</code> starts a fresh
              conversation.
            </li>
          </ul>
        </div>
      </div>
      <button
        onClick={logout}
        className="border-t border-bw-border px-4 py-3 text-left text-xs text-neutral-400 hover:bg-bw-bg hover:text-neutral-100"
      >
        Sign out
      </button>
    </aside>
  );
}
