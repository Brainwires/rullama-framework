# brainclaw-mcp-google-chat

Google Chat channel adapter for the BrainClaw gateway.

Ingests events via a Google-signed HTTPS webhook and posts replies via
the Chat REST API using an OAuth bearer minted from a bot service-account
key.

## Setup

1. Create a Google Cloud project; enable the Google Chat API.
2. Create a service account dedicated to the bot. Grant it the
   **Chat Bot** IAM role (or equivalent). Download the JSON key.
3. In the Chat API console, configure the bot to publish events to an
   **HTTPS endpoint** pointing at `https://<your-host>/events`.
4. Set the **audience** on the endpoint to a stable string you control
   (e.g. `https://chat.example.com/brainclaw`). This becomes the
   expected `aud` JWT claim.

## Configuration (env / flags)

| Variable | Flag | Purpose |
|---|---|---|
| `GOOGLE_CHAT_PROJECT_ID` | `--project-id` | GCP project id |
| `GOOGLE_CHAT_AUDIENCE` | `--audience` | Expected `aud` on ingress JWTs |
| `GOOGLE_CHAT_SERVICE_ACCOUNT_KEY` | `--service-account-key` | Path to JSON key |
| `GATEWAY_URL` | `--gateway-url` | WS URL of `brainwires-gateway` |
| `GATEWAY_TOKEN` | `--gateway-token` | Optional handshake token |
| `LISTEN_ADDR` | `--listen-addr` | Default `0.0.0.0:9101` |

## Run modes

```bash
# Gateway + webhook mode (default).
brainclaw-mcp-google-chat serve

# Also expose MCP stdio tools for ad-hoc scripting.
brainclaw-mcp-google-chat serve --mcp

# MCP-only (no webhook, no gateway). Uses stdio transport.
brainclaw-mcp-google-chat mcp --service-account-key /path/to/key.json

# Version info.
brainclaw-mcp-google-chat version
```

## Ingress event handling

- `MESSAGE` – forwarded as user message (`argumentText` preferred).
- `CARD_CLICKED` – forwarded as user message whose text is the action id.
- `ADDED_TO_SPACE` / `REMOVED_FROM_SPACE` – logged, not forwarded.

All requests are rejected with 401 unless the `Authorization` header
carries a valid RS256 JWT signed by Google, with `aud` matching the
configured audience and a non-expired `exp`.

## Capabilities

`RICH_TEXT | MENTIONS | THREADS` — no reactions (bot API doesn't expose
them), no typing indicators (not supported server-side).

## Security notes

- OAuth bearer tokens are cached until five minutes before expiry, then
  re-minted from the service-account assertion.
- JWKs are cached for one hour (matching Google's rotation cadence).
- Inbound user ids are SHA-256 truncated before being written to the
  audit log; message bodies never appear in logs.
