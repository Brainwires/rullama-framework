# brainclaw-mcp-line

LINE Messaging API channel adapter for BrainClaw.

## How it works

Inbound: LINE POSTs events to `/webhook` on the configured listen
address. Each request is signed with `X-Line-Signature: <base64(HMAC-SHA256(body, channel_secret))>`;
the adapter verifies the signature and rejects any mismatch with 401.

Outbound: the adapter prefers LINE's `reply` endpoint when a fresh
(<55s old) reply token is cached from a recent inbound event, falling
back to the `push` endpoint otherwise. Reply tokens are single-use and
expire quickly, so the cache is drained on every successful reply.

## Setup

1. Create a LINE channel in the LINE Developers console.
2. Note the **Channel secret** (for signature verification) and issue a
   long-lived **Channel access token** (for outbound API calls).
3. Point the channel's webhook URL at `https://<your-host>/webhook` —
   TLS termination is your responsibility (Caddy, Nginx, etc.).

## Configuration

| Variable | Flag | Purpose |
|---|---|---|
| `LINE_CHANNEL_SECRET` | `--channel-secret` | HMAC key for signature verification |
| `LINE_CHANNEL_ACCESS_TOKEN` | `--channel-access-token` | Outbound bearer token |
| `GATEWAY_URL` | `--gateway-url` | WS URL of `brainwires-gateway` |
| `GATEWAY_TOKEN` | `--gateway-token` | Optional handshake token |
| `LISTEN_ADDR` | `--listen-addr` | Default `0.0.0.0:9104` |

## Run modes

```bash
brainclaw-mcp-line serve
brainclaw-mcp-line serve --mcp
brainclaw-mcp-line mcp
brainclaw-mcp-line version
```

## Event handling

- `message` type `text` — forwarded as user message.
- `message` types `image | video | audio | file` — forwarded as a short
  marker such as `[image attachment id=42]`. Fetching raw content
  requires an extra authenticated call and is intentionally deferred to
  the MCP tool layer for MVP.
- `postback` — button clicks; the `data` field is forwarded as text.
- `follow`, `unfollow`, `join`, `leave`, `memberJoined`, `memberLeft` —
  logged and dropped.

## Capabilities

`RICH_TEXT | MENTIONS`. No threads, no reactions (LINE's bot API does
not expose reaction authoring), attachments surface as URL/marker
references only.

## Security notes

- Missing or invalid `X-Line-Signature` → 401 with a generic message.
  The signature itself is never logged.
- User ids are SHA-256 truncated in audit lines.
- Reply tokens older than 55s are silently dropped — the adapter
  transparently falls back to push.
