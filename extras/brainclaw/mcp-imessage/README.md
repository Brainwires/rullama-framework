# brainclaw-mcp-imessage

iMessage channel adapter for BrainClaw, bridged through a local
[BlueBubbles](https://bluebubbles.app) server running on a Mac.

## How it works

The adapter polls the BlueBubbles REST API on a configurable interval,
converts new inbound messages into `ChannelEvent::MessageReceived`, and
forwards them to the Brainwires gateway over a WebSocket. Outbound
messages from the agent are POSTed back to BlueBubbles'
`/api/v1/message/text` endpoint.

A per-chat last-seen GUID is persisted under `state_dir`
(`~/.brainclaw/state/imessage.json` by default) so restarts never
re-forward an already-seen message. Writes are atomic-rename.

## Setup

1. Install BlueBubbles Server on the Mac that holds your iMessage
   account, configure a password, and expose its HTTP API to this host
   (Tailscale, ngrok, or similar — the adapter does not care how).
2. Export the following variables before running:

| Variable | Flag | Purpose |
|---|---|---|
| `BB_SERVER_URL` | `--server-url` | e.g. `https://mac.tailnet.ts.net:1234` |
| `BB_PASSWORD` | `--password` | BlueBubbles server password |
| `BB_POLL_INTERVAL_SECS` | `--poll-interval-secs` | Default `2` |
| `BB_CHATS` | `--chats` | Comma-separated chat GUIDs (empty = all) |
| `BB_STATE_DIR` | `--state-dir` | Cursor store (default `~/.brainclaw/state`) |
| `GATEWAY_URL` | `--gateway-url` | WS URL of `brainwires-gateway` |
| `GATEWAY_TOKEN` | `--gateway-token` | Optional handshake token |

## Run modes

```bash
brainclaw-mcp-imessage serve              # polling + gateway
brainclaw-mcp-imessage serve --mcp        # + stdio MCP tools
brainclaw-mcp-imessage mcp                # stdio MCP only
brainclaw-mcp-imessage version
```

## Capabilities

`REACTIONS | DELETE_MESSAGES` (tapbacks are supported via the MCP
`react` tool). No typing indicators, no threads, attachments are
surfaced as URL references only.

## Security notes

- The server password is appended as `?password=` on every request and
  is never logged.
- Only messages where `isFromMe = false` are forwarded.
- Handles (phone numbers / emails) are SHA-256 truncated when written
  to audit lines; full values stay in metadata for in-process use.
