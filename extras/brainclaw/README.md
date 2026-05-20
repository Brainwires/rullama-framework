# BrainClaw

Personal AI assistant daemon built on the [Brainwires Framework](https://github.com/Brainwires/brainwires-framework).

## Overview

BrainClaw is a multi-channel, always-on AI assistant that connects your messaging platforms — Discord, Slack, Telegram, WhatsApp, Matrix, Mattermost, Signal, GitHub — to a persistent agent session via a central gateway. Each channel runs as a standalone MCP server; the gateway routes messages between them and the agent.

## Architecture

```
Discord ──┐
Slack ────┤
Telegram ─┤  (MCP channel servers)
WhatsApp ─┼──▶  brainwires-gateway  ──▶  brainclaw daemon  ──▶  AI provider
Matrix ───┤                                    │
GitHub ───┤                           skill-registry MCP
Signal ───┘
```

## Components

| Crate | Binary | Description |
|---|---|---|
| `daemon` | `brainclaw` | Core daemon — agent session, skill dispatch, cron scheduling |
| `gateway` | `brainwires-gateway` | WebSocket gateway routing messages between channel servers and the daemon |
| `mcp-discord` | `mcp-discord` | Discord channel adapter |
| `mcp-slack` | `mcp-slack` | Slack channel adapter (Socket Mode) |
| `mcp-telegram` | `mcp-telegram` | Telegram channel adapter |
| `mcp-whatsapp` | `mcp-whatsapp` | WhatsApp Business adapter (Meta Graph API) |
| `mcp-matrix` | `mcp-matrix` | Matrix rooms adapter (matrix-sdk) |
| `mcp-mattermost` | `mcp-mattermost` | Mattermost adapter |
| `mcp-signal` | `mcp-signal` | Signal adapter (signal-cli REST API) |
| `mcp-github` | `mcp-github` | GitHub webhooks + operations adapter |
| `mcp-google-chat` | `brainclaw-mcp-google-chat` | Google Chat bot adapter (HTTPS webhook + Chat REST API) |
| `mcp-teams` | `brainclaw-mcp-teams` | Microsoft Teams adapter (Bot Framework ingress/egress) |
| `mcp-irc` | `brainclaw-mcp-irc` | IRC adapter (persistent TCP, TLS/SASL) |
| `mcp-imessage` | `brainclaw-mcp-imessage` | iMessage adapter via the BlueBubbles REST bridge |
| `mcp-nextcloud-talk` | `brainclaw-mcp-nextcloud-talk` | Nextcloud Talk (Spreed) adapter |
| `mcp-line` | `brainclaw-mcp-line` | LINE Messaging API adapter (HMAC webhook + REST egress) |
| `mcp-feishu` | `brainclaw-mcp-feishu` | Feishu / Lark adapter (signed webhook + tenant-token egress) |
| `mcp-skill-registry` | `mcp-skill-registry` | Skill marketplace — stores, searches, and serves distributable skill packages |

## Quick start

```toml
# In your shell config or systemd unit
BRAINCLAW_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-…
BRAINCLAW_DISCORD_TOKEN=…
```

```sh
cargo build --release -p brainclaw -p brainwires-gateway -p mcp-discord

./target/release/brainwires-gateway &
./target/release/mcp-discord &
./target/release/brainclaw
```

## Feature flags

| Flag | Description |
|---|---|
| `native-tools` *(default)* | Enable native filesystem and shell tools |
| `voice` | Enable microphone/speaker I/O via `brainwires-hardware` |
| `email` | Email tool support |
| `calendar` | Calendar tool support |
| `rag` | RAG retrieval tool support |
| `browser` | Browser automation tool support |

## License

MIT OR Apache-2.0
