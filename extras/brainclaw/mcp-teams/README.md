# brainclaw-mcp-teams

Microsoft Teams channel adapter for the BrainClaw gateway.

Implements the Bot Framework ingress protocol: Microsoft POSTs Activity
JSON to `/api/messages`, authenticated via a JWT from the Bot Framework
OpenID metadata. Replies are POSTed back to the `serviceUrl` embedded in
each inbound Activity, using an OAuth client-credentials bearer.

## Setup

1. Register an Azure AD app (multi-tenant or single-tenant).
2. Create a **Bot Channels Registration** in Azure pointing at
   `https://<your-host>/api/messages`.
3. Install the Microsoft Teams channel on the bot.
4. Save the app's **application (client) id** and **client secret** —
   those become `TEAMS_APP_ID` and `TEAMS_APP_PASSWORD`.

## Configuration

| Variable | Flag | Purpose |
|---|---|---|
| `TEAMS_APP_ID` | `--app-id` | Azure AD client id (bot app id) |
| `TEAMS_APP_PASSWORD` | `--app-password` | Azure AD client secret |
| `TEAMS_TENANT_ID` | `--tenant-id` | `common` for multi-tenant, else GUID |
| `GATEWAY_URL` | `--gateway-url` | Gateway WS URL |
| `GATEWAY_TOKEN` | `--gateway-token` | Optional handshake token |
| `LISTEN_ADDR` | `--listen-addr` | Default `0.0.0.0:9102` |

## Run modes

```bash
# Default: webhook + gateway.
brainclaw-mcp-teams serve

# With MCP stdio tools alongside.
brainclaw-mcp-teams serve --mcp

# MCP-only.
brainclaw-mcp-teams mcp --app-id ... --app-password ... --tenant-id common

# Version.
brainclaw-mcp-teams version
```

## Activity type mapping

- `message` — forwarded as user message.
- `invoke` (adaptive-card action) — forwarded as user message whose text
  is the JSON-encoded action payload.
- `conversationUpdate` — logged; also records the `serviceUrl` needed for
  future replies.
- `typing` — dropped.

All activities are rejected with 401 if the JWT fails verification. We
accept tokens from any of Microsoft's clouds (public, GCC, DoD) because
the signing JWKs are scoped by issuer via the OIDC metadata document.

## Capabilities

`RICH_TEXT | MENTIONS | THREADS` — Teams reactions and message
edit/delete require extra scopes and are not implemented at MVP.

## Known limits

- Replies only work for conversations that have seen at least one
  inbound activity (the Bot Framework `serviceUrl` must be recorded
  first). Entries expire after 24h.
- Message history is not fetched — Teams requires the Graph API and a
  separate set of permissions.
