#!/usr/bin/env node
/**
 * Minimal smoke test for the brainwires-webchat Next.js app.
 *
 * Boots `next start` on port 3101, then hits:
 *   - GET  /login          → expects 200 and the login form HTML
 *   - GET  /api/token      → expects 401 (no cookie)
 *   - POST /api/auth       → expects 200 in dev mode (empty
 *     WEBCHAT_ADMIN_TOKEN) and a set-cookie header
 *
 * Fails fast with a non-zero exit code on any assertion failure.
 *
 * Designed to be runnable against the output of `pnpm build`.
 */

import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const PORT = 3101;
const BASE = `http://127.0.0.1:${PORT}`;

function log(msg) {
  console.log(`[smoke] ${msg}`);
}

async function waitForPort(timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(BASE + "/login", { redirect: "manual" });
      if (res.status > 0) return;
    } catch {
      // still starting
    }
    await delay(200);
  }
  throw new Error(`server did not start within ${timeoutMs}ms`);
}

async function main() {
  const env = {
    ...process.env,
    PORT: String(PORT),
    WEBCHAT_SECRET: "smoke-test-secret",
    WEBCHAT_ADMIN_TOKEN: "",
    NEXT_PUBLIC_GATEWAY_WS: "ws://localhost:18789",
  };

  log("spawning `next start`…");
  const child = spawn(
    "node",
    ["node_modules/next/dist/bin/next", "start", "-p", String(PORT)],
    { env, stdio: ["ignore", "pipe", "inherit"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(`[next] ${chunk}`));

  let exitCode = 0;
  try {
    await waitForPort(30_000);
    log("server ready");

    {
      const res = await fetch(BASE + "/login", { redirect: "manual" });
      if (res.status !== 200) {
        throw new Error(`/login returned ${res.status}`);
      }
      const body = await res.text();
      if (!body.includes("Admin Token") && !body.includes("BrainClaw")) {
        throw new Error("/login body did not contain expected markers");
      }
      log("GET /login OK");
    }

    {
      const res = await fetch(BASE + "/api/token", { redirect: "manual" });
      if (res.status !== 401) {
        throw new Error(`/api/token expected 401, got ${res.status}`);
      }
      log("GET /api/token (anonymous) → 401 OK");
    }

    {
      const res = await fetch(BASE + "/api/auth", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ admin_token: "anything" }),
        redirect: "manual",
      });
      if (res.status !== 200) {
        throw new Error(`/api/auth expected 200, got ${res.status}`);
      }
      const cookie = res.headers.get("set-cookie");
      if (!cookie || !cookie.includes("webchat_jwt=")) {
        throw new Error("/api/auth did not set webchat_jwt cookie");
      }
      log("POST /api/auth → 200 + cookie OK");
    }

    log("all checks passed");
  } catch (err) {
    console.error(`[smoke] FAILED: ${err?.message ?? err}`);
    exitCode = 1;
  } finally {
    child.kill("SIGTERM");
  }

  process.exit(exitCode);
}

void main();
