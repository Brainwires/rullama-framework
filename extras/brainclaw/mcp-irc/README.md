# brainclaw-mcp-irc

IRC channel adapter for the BrainClaw gateway.

Maintains a persistent (optionally TLS) connection to a single IRC
network, joins configured channels, and bridges PRIVMSG traffic between
IRC and the gateway. Outbound replies are chunked to stay within IRC's
512-byte line limit.

## Setup

1. Pick an IRC network (e.g. Libera Chat).
2. Register your bot's nick with NickServ if required.
3. Export the env vars below (or pass them via CLI flags).

## Configuration

| Variable | Flag | Default | Purpose |
|---|---|---|---|
| `IRC_SERVER` | `--server` | `irc.libera.chat` | Hostname |
| `IRC_PORT` | `--port` | `6697` | TCP port |
| `IRC_USE_TLS` | `--use-tls` | `true` | TLS on |
| `IRC_NICK` | `--nick` |  | Bot nick (required) |
| `IRC_USERNAME` | `--username` | `brainclaw` | USER field |
| `IRC_REALNAME` | `--realname` | `BrainClaw Bot` | GECOS |
| `IRC_SASL_PASSWORD` | `--sasl-password` |  | Optional SASL PLAIN pass |
| `IRC_CHANNELS` | `--channels` |  | Comma-separated `#foo,#bar` |
| `IRC_MESSAGE_PREFIX` | `--message-prefix` | `brainclaw: ` | Forward filter |
| `GATEWAY_URL` | `--gateway-url` | local | Gateway WS URL |
| `GATEWAY_TOKEN` | `--gateway-token` |  | Gateway handshake token |

## Forwarding rules

- **PMs to the bot** are always forwarded.
- **Channel messages** are only forwarded when they start with
  `IRC_MESSAGE_PREFIX`. The prefix is stripped before forwarding.
- **CTCP ACTION** (`/me`) frames are preserved — forwarded as
  `*nick did something*`.

## Run modes

```bash
# Default: IRC + gateway.
brainclaw-mcp-irc serve --nick mybot --channels '#botdev'

# Also expose MCP stdio tools.
brainclaw-mcp-irc serve --nick mybot --channels '#botdev' --mcp

# MCP-only (still connects to IRC so `send_message` has a live socket).
brainclaw-mcp-irc mcp --server irc.libera.chat --nick mybot

# Version.
brainclaw-mcp-irc version
```

## Nick-collision handling

If the server rejects the configured nick with ERR_NICKNAMEINUSE (433),
the adapter retries exactly once with a trailing underscore (`mybot_`)
and emits a WARN log. After that, the connection loop backs off
exponentially (2s → 60s cap).

## Capabilities

`ChannelCapabilities::empty()` — plain text only. No rich text, no
reactions, no threads, no attachments. Outbound messages are
UTF-8-safely chunked at 400 bytes per PRIVMSG.
