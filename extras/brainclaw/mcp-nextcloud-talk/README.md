# brainclaw-mcp-nextcloud-talk

Nextcloud Talk (Spreed) channel adapter for BrainClaw.

## How it works

The adapter polls the Nextcloud Talk OCS REST API for each configured
room token, converting new messages into
`ChannelEvent::MessageReceived` and forwarding them to the Brainwires
gateway. Outbound messages are POSTed back to the same REST endpoint
using form-encoded bodies.

A per-room `last_message_id` is persisted under `state_dir`
(`~/.brainclaw/state/nextcloud_talk.json` by default) with atomic-rename
writes so restarts don't re-forward messages.

## Setup

1. In Nextcloud, create an **app password** for the bot user
   (Settings → Security → Create new app password). Never use the
   account password.
2. Find the room tokens (the opaque id in the URL when you open a
   Talk conversation in the web UI).
3. Export the variables below.

| Variable | Flag | Purpose |
|---|---|---|
| `NEXTCLOUD_URL` | `--server-url` | e.g. `https://cloud.example.com` |
| `NEXTCLOUD_USERNAME` | `--username` | Nextcloud user id (not email) |
| `NEXTCLOUD_APP_PASSWORD` | `--app-password` | App password |
| `NEXTCLOUD_ROOMS` | `--rooms` | Comma-separated room tokens |
| `NEXTCLOUD_POLL_INTERVAL_SECS` | `--poll-interval-secs` | Default `2` |
| `NEXTCLOUD_STATE_DIR` | `--state-dir` | Cursor store |
| `GATEWAY_URL` | `--gateway-url` | WS URL of `brainwires-gateway` |
| `GATEWAY_TOKEN` | `--gateway-token` | Optional handshake token |

## Run modes

```bash
brainclaw-mcp-nextcloud-talk serve
brainclaw-mcp-nextcloud-talk serve --mcp
brainclaw-mcp-nextcloud-talk mcp
brainclaw-mcp-nextcloud-talk version
```

## API quirks

- Every request must carry `OCS-APIRequest: true` — without it Nextcloud
  returns 406. We set it automatically on every call.
- Session ids are shaped as `nextcloud:<host>:<room_token>:<user_id>`.

## Capabilities

`RICH_TEXT | MENTIONS | THREADS | DELETE_MESSAGES`. Thread parents are
read (parent message id surfaces as `reply_to` + `thread_id`) but
authoring threaded replies with reply-to ids is only exposed via the
MCP `send_message` tool.

## Security notes

- App passwords are converted to an in-memory `Basic` header at
  construction; they are never logged.
- User ids are SHA-256 truncated when written to audit lines; full ids
  only live in per-message metadata.
