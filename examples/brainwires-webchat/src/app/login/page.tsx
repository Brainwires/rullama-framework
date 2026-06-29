"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/auth", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ admin_token: token }),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `login failed (${res.status})`);
      }
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm space-y-4 rounded-lg border border-bw-border bg-bw-surface p-6 shadow-xl"
      >
        <div>
          <h1 className="text-xl font-semibold text-bw-accent">BrainClaw Chat</h1>
          <p className="mt-1 text-sm text-neutral-400">
            Paste your admin token to mint a chat session.
          </p>
        </div>
        <label className="block">
          <span className="block text-xs uppercase tracking-wide text-neutral-400">
            Admin Token
          </span>
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            autoComplete="off"
            autoFocus
            className="mt-1 w-full rounded border border-bw-border bg-bw-bg px-3 py-2 text-sm text-neutral-100 outline-none focus:border-bw-accent"
          />
        </label>
        {error ? (
          <div className="rounded border border-red-600/40 bg-red-900/20 px-3 py-2 text-xs text-red-200">
            {error}
          </div>
        ) : null}
        <button
          type="submit"
          disabled={submitting || token.length === 0}
          className="w-full rounded bg-bw-accent px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
