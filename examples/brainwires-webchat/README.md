# brainwires-webchat

Browser chat client for the BrainClaw daemon.

This is a Next.js 15 (App Router) application that talks to the
BrainClaw gateway's JWT-gated `/webchat/ws` endpoint. It intentionally
lives outside the `extras/brainwires-docs/` app so the two can be
deployed independently.

## Quick start

```bash
pnpm install
cp .env.example .env.local
# edit .env.local — at minimum set WEBCHAT_SECRET to match your gateway
pnpm dev   # → http://localhost:3001
```

The dev server defaults to **port 3001** to avoid the `brainwires-docs`
Studio server which owns 3000.

## Environment variables

See `.env.example`.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_GATEWAY_WS` | yes | `ws://localhost:18789` | Base URL of the gateway's WebSocket; `/webchat/ws` is appended. |
| `NEXT_PUBLIC_GATEWAY_HTTP` | no | `http://localhost:18789` | Base URL of the gateway HTTP API. |
| `WEBCHAT_SECRET` | yes | — | HS256 secret used to sign browser JWTs. Must match the gateway's `[webchat] jwt_secret`. |
| `WEBCHAT_ADMIN_TOKEN` | no | `""` | Admin token the `/login` page accepts; empty means dev-mode (accept any non-empty token). |

## Scripts

```bash
pnpm dev        # Next.js dev server on port 3001
pnpm build      # production build
pnpm start      # serve the production build on port 3001
pnpm lint       # ESLint
pnpm typecheck  # tsc --noEmit
pnpm test:smoke # boots `next start` on 3101 and runs HTTP smoke checks
```

## Architecture

- `src/app/login/page.tsx` — single-field token form. POSTs to `/api/auth`.
- `src/app/api/auth/route.ts` — validates the admin token, mints an HS256
  JWT, sets it as an HttpOnly cookie.
- `src/app/api/token/route.ts` — same-origin endpoint that returns the
  raw JWT so the browser can attach it as `?token=` to the gateway
  WebSocket URL (HttpOnly cookies cannot be attached to cross-origin
  WebSocket handshakes from `new WebSocket(url)`).
- `src/app/page.tsx` — server component that enforces the cookie check
  and, if valid, renders the chat UI.
- `src/components/ChatPane.tsx` — message list, streaming support,
  history backfill on reconnect.
- `src/lib/ws.ts` — WebSocket client with exponential-backoff reconnect.
- `src/lib/jwt.ts` — server-only HS256 sign/verify helpers.

## Protocol

The WebSocket wire protocol is documented in the gateway source at
`extras/brainclaw/gateway/src/webchat.rs`. In short:

```jsonc
// client -> server
{ "type": "message", "content": "hi" }
{ "type": "resume",  "session_id": "webchat:<user>" }

// server -> client
{ "type": "session", "id": "webchat:<user>" }
{ "type": "chunk",   "content": "partial" }
{ "type": "message", "role": "assistant", "content": "final", "id": "uuid" }
{ "type": "error",   "message": "reason" }
```

## Scope

This initial cut delivers login, a single persistent chat session,
reconnect-with-history, and slash-command autocomplete. The
multi-session sidebar, attachment upload, and per-frame tool-use
previews are deliberately scoped out of v1 and tracked as TODOs in
`CLAUDE.md` on the BrainClaw framework.
