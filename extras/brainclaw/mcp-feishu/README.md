# brainclaw-mcp-feishu

Feishu / Lark channel adapter for BrainClaw.

## How it works

Inbound events arrive on `/webhook` and are authenticated via Feishu's
custom HMAC-SHA256 scheme:

```text
signature = base64( HMAC-SHA256( timestamp || nonce || body, verification_token ) )
```

sent in `X-Lark-Signature` alongside `X-Lark-Request-Timestamp` and
`X-Lark-Request-Nonce`. Signature mismatches return 401.

The onboarding `url_verification` challenge is answered without a
session — we echo back the `challenge` field as required by the spec.

Outbound messages POST to `/open-apis/im/v1/messages` using a tenant
access token minted from the configured `app_id + app_secret`. The
token is cached until 5 minutes before expiry.

## Setup

1. Create a Feishu / Lark app in the Open Platform console.
2. Note the **App ID**, **App Secret**, and **Verification Token**.
3. Point the event subscription URL at `https://<your-host>/webhook`.
4. (Optional) If you configure AES encryption in the console, set
   `FEISHU_ENCRYPT_KEY`; the adapter logs a warning and expects events
   in the clear at MVP — disable encryption in the console for now.

## Configuration

| Variable | Flag | Purpose |
|---|---|---|
| `FEISHU_APP_ID` | `--app-id` | App id |
| `FEISHU_APP_SECRET` | `--app-secret` | App secret |
| `FEISHU_VERIFICATION_TOKEN` | `--verification-token` | HMAC key for signature verification |
| `FEISHU_ENCRYPT_KEY` | `--encrypt-key` | Optional AES key (MVP: warn-and-ignore) |
| `GATEWAY_URL` | `--gateway-url` | WS URL of `brainwires-gateway` |
| `GATEWAY_TOKEN` | `--gateway-token` | Optional handshake token |
| `LISTEN_ADDR` | `--listen-addr` | Default `0.0.0.0:9105` |

## Run modes

```bash
brainclaw-mcp-feishu serve
brainclaw-mcp-feishu serve --mcp
brainclaw-mcp-feishu mcp
brainclaw-mcp-feishu version
```

## Event handling

- `im.message.receive_v1` — user chat message. `text` and `post`
  (rich) bodies are flattened to text; `image`/`file`/`audio` become
  markers like `[image]`.
- `card.action.trigger` — button click on an interactive card;
  forwarded as the action tag text.
- `im.message.message_read_v1` — dropped.
- `url_verification` — handshake, answered inline.

## Capabilities

`RICH_TEXT | MENTIONS | THREADS`. Feishu cards are supported for
outbound plain-text egress; richer rendering is deferred.

## Security notes

- Tenant tokens are cached in memory only; they are never logged.
- User ids (`ou_*`) are SHA-256 truncated in audit lines.
- Signature mismatches always return a generic 401 with no detail.
