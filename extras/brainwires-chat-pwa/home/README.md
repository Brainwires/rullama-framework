# brainwires-home

Dial-home daemon for the [Brainwires chat PWA](../web/). Runs on the user's
own hardware and exposes a WebRTC peer + A2A JSON-RPC bridge into
[`brainwires-agent::TaskAgent`](../../../crates/brainwires-agent/), so the
PWA in a browser can talk to a powerful agent on the user's home machine
without paying anyone a per-token markup.

The PWA reaches this daemon via WebRTC behind a Cloudflare Tunnel (or
equivalent). The tunnel only ever forwards HTTPS signaling (`/signal/*`,
`/pair/*`, `/.well-known/agent-card.json`) — once the WebRTC peer is
negotiated, A2A traffic flows peer-to-peer over an SCTP DataChannel.

## Status

**Phase 2 complete.** All 12 milestones landed on the version branch.

| Milestone | What lands                                                              | Status     |
|-----------|--------------------------------------------------------------------------|------------|
| M1        | Crate scaffold, `webrtc-rs` dep, in-process two-peer ping/pong test     | landed     |
| M2        | Axum `/signal/*` endpoints, in-memory session map, agent-card JSON      | landed     |
| M3        | Wire the WebRTC peer into the axum routes; JSON-RPC `ping` echo         | landed     |
| M4        | A2A bridge — route inbound JSON-RPC into a real `TaskAgent`             | landed     |
| M5        | Browser-side dial-home transport (`web/src/home-*`)                     | landed     |
| M6        | CORS configuration + tunnel-pointing docs                                | landed     |
| M7        | Cloudflare Calls TURN credential minting (cellular symmetric-NAT path)  | landed     |
| M8        | Pairing flow (`/pair/claim`, `/pair/confirm`, QR + 6-digit confirm)     | landed     |
| M9        | `home-provider.js` adapter — "Home agent" appears as a chat provider   | landed     |
| M10       | Reconnect/resume — heartbeat, ICE restart, outbox replay                | landed     |
| M11       | Multimodal chunking — `bin/begin` + `bin/chunk` + `bin/end`            | landed     |
| **M12**   | Polish — connection-status pill, error toasts, "remove paired device"  | landed     |

## Architecture

```
+-----------------+       HTTPS (signaling)        +----------------------+
|  PWA (browser)  | ---> POST /signal/offer  --->  |  Cloudflare Tunnel   |
|  vanilla JS     | <--- GET  /signal/answer ----- |  cloudflared         |
|  RTCPeer...     | <--- GET  /signal/ice    ----- |  (trycloudflare or   |
+--------+--------+                                |   user-owned domain) |
         |                                         +----------+-----------+
         |  WebRTC SCTP DataChannel ("a2a")                   |
         |  (Cloudflare Calls TURN if needed)                 v
         |                                          +-------------------+
         +----------- ICE/DTLS/SCTP --------------> | brainwires-home   |
                                                    | axum :7878        |
                                                    | signaling+WebRTC  |
                                                    +---------+---------+
                                                              | A2A JSON-RPC
                                                              v
                                                  TaskAgent / AgentPool /
                                                  Providers / MCP / files
```

## Quickstart

```sh
cargo run -p brainwires-home -- --help
cargo run -p brainwires-home -- --bind 127.0.0.1:7878
```

The default bind is `127.0.0.1:7878` — the tunnel sits in front of it. The
daemon never needs to listen on a public interface; if it does, you've
mis-configured the tunnel.

## Endpoints

### Signaling (M2 — wired)

| Method | Path                        | Body / response                                     | Status |
|--------|-----------------------------|-----------------------------------------------------|--------|
| POST   | `/signal/session`           | → `{ session_id, ice_servers: [...] }`              | M2     |
| POST   | `/signal/offer/{session}`   | `{ sdp, type }` → `204`                             | M2     |
| GET    | `/signal/answer/{session}`  | long-poll 25 s → `{ sdp, type }` or `204`           | M2     |
| POST   | `/signal/ice/{session}`     | `{ candidate, sdpMid, sdpMLineIndex }` → `204`      | M2     |
| GET    | `/signal/ice/{session}?since=N` | long-poll → `{ candidates: [...], cursor }`     | M2     |
| DELETE | `/signal/{session}`         | → `204` (idempotent)                                | M2     |

State is **in-memory only**: an `Arc<DashMap<String, Arc<SessionState>>>` held
in axum `State`. A background task GCs sessions older than 30 minutes every
60 s. Long-poll handlers `await` a `tokio::sync::Notify` with a deadline; on
timeout `/signal/answer` returns `204` (PWA retries) and `/signal/ice` returns
the current snapshot.

`ice_servers` always carries at least the two public STUN fallbacks
(`stun.cloudflare.com:3478`, `stun.l.google.com:19302`). With Cloudflare
Calls TURN configured, a freshly-minted ~10 min credential is prepended.
See § TURN credentials below.

### Well-known (M2 — wired)

| Method | Path                              | Description                          |
|--------|-----------------------------------|--------------------------------------|
| GET    | `/.well-known/agent-card.json`    | A2A 0.3 AgentCard for discovery      |

The card is built from `brainwires_a2a::AgentCard`, advertises `streaming:
true`, `defaultInputModes: ["text"]`, and a single `JSONRPC`
`supportedInterfaces` entry pinned to `A2A_PROTOCOL_VERSION` ("0.3").

### Pairing (M8 — wired)

| Method | Path              | Body / response                                                                  |
|--------|-------------------|----------------------------------------------------------------------------------|
| POST   | `/pair/claim`     | `{ one_time_token, device_pubkey, device_name }` → `200 { ok: true }` / `404`   |
| POST   | `/pair/confirm`   | `{ one_time_token, code }` → `200 { device_token, cf_client_id?, cf_client_secret?, peer_pubkey }` / `401` (wrong code) / `404` (expired/unknown) / `400` (no claim) |

Pending offers expire after **5 minutes**. A wrong-code submit consumes the
offer (single-shot) — the operator can run `brainwires-home pair` again to
mint a fresh one. Confirmed device records are persisted to
`~/.brainwires/home/devices.json` (atomic write, `0600` on Unix). The
daemon's stable identity pubkey is generated on first start at
`~/.brainwires/home/identity.json` (`0600`); the same value is returned in
every `peer_pubkey` so the PWA can pin it (TOFU).

### CLI: `brainwires-home pair`

```sh
cargo run -p brainwires-home -- pair --tunnel-url https://home.example.com
```

Mints one offer, prints the `bwhome://pair?u=…&t=…&fp=…` URL plus a 6-digit
confirm code, and waits up to 5 minutes for the PWA to claim + confirm.
On success, prints the resulting `device_name` / `device_pubkey` /
`granted_at` and exits.

The QR URL is plain text — render it through any QR encoder (we don't
ship one in the daemon). On most home machines the easiest path is
`qrencode -t ANSIUTF8 'bwhome://...'` piped from the daemon's stdout.

### CF Access (optional)

If the operator runs the daemon behind Cloudflare Access, they can pre-
provision a service-token pair and pass it via env / flags:

```sh
brainwires-home --cf-access-client-id <ID> --cf-access-client-secret <SECRET> ...
```

The same `(client_id, client_secret)` is returned alongside `device_token`
on every successful pairing — the PWA includes it as
`CF-Access-Client-Id` / `CF-Access-Client-Secret` headers on signaling
requests. CF Access is **optional**; without it the daemon validates only
the `Authorization: Bearer <device_token>` (the primary auth gate).

## Using the home agent from the chat PWA (M9)

Once paired (M8), the chat UI lists **"Home agent"** as a provider alongside
the cloud (Anthropic / OpenAI / Google) and local (Gemma 4 E2B) options.
Tap the provider chip in the composer to cycle to it; messages then route
over the WebRTC data channel through the daemon's A2A bridge into
`brainwires-agent::ChatAgent`.

The PWA-side code is in
[`web/src/home-provider.js`](../web/src/home-provider.js). It implements
the `EventProvider` interface: a single `message/send` round-trip per
turn, with the reply dispatched as a `chat_chunk` + `chat_done` pair on
`state.events` — the same channel cloud / local providers use, so the
chat UI is provider-agnostic.

The home provider currently ships **non-streaming** (one chunk per
turn) — incremental streaming via `message/stream` is a deferred
follow-up. See [Known limitations](#known-limitations) below.

## Data-channel protocol — A2A JSON-RPC over SCTP

Frame: `[u32 LE length][JSON bytes]`. SCTP is ordered + reliable; we do not
reinvent retransmit/sequence. Payloads are
[`brainwires-a2a`](../../../crates/brainwires-a2a/) `JsonRpcRequest` /
`JsonRpcResponse` / `message/stream` partial-result envelopes verbatim —
zero new schema. Streaming tokens are one frame per datachannel send.

### Binary chunking (M11)

Payloads larger than the SCTP frame budget (or just larger than 64 KB,
where inlining stops being free) ride a three-call JSON-RPC sequence
instead of one big `message/send`:

| Method      | Params                                                          | Reply                          |
|-------------|-----------------------------------------------------------------|--------------------------------|
| `bin/begin` | `{ bin_id, content_type, total_size, total_chunks }`            | `{ ok: true }`                 |
| `bin/chunk` | `{ bin_id, seq, data: <base64> }`                               | `{ ok: true }`                 |
| `bin/end`   | `{ bin_id, sha256? }`                                           | `{ ok: true, size }`           |

Default raw chunk size: **256 KB** (≈341 KB after base64). The PWA
generates `bin_id` as a UUID and uploads sequentially. The home daemon
keeps pending buffers per session for 30 s and finalized blobs for 5
minutes; both are GC'd by the existing 60 s tick. A subsequent
`message/send` whose `parts[]` contains an entry with
`metadata.bin_id == "<id>"` consumes the blob — the daemon strips the
metadata and inlines the bytes as `Part.raw` (base64) before forwarding
to the agent. One-shot: a second `message/send` referencing the same
`bin_id` finds nothing.

Custom error codes (in addition to the spec range):

| Code     | Meaning                                                |
|----------|--------------------------------------------------------|
| `-32001` | unknown `bin_id` (chunk before begin, or after expiry) |
| `-32002` | `seq` doesn't match `next_expected`                    |
| `-32003` | sha256 mismatch on `bin/end` (buffer is dropped)       |

Lives in `home/src/binary.rs` (the store + parsing) and `home/src/webrtc.rs`
(JSON-RPC dispatch + the `message/send` rewrite that resolves bin refs
to inline bytes).

## Reconnect

15 s app-level ping. On `iceconnectionstate === "disconnected"` for >5 s,
the PWA initiates an ICE restart on the same signaling endpoints. A new
`session_id` is only minted on a second restart failure.

The home daemon keeps a bounded ring buffer of the last 64 messages by
JSON-RPC `id`. On reconnect the PWA sends `resume { last_seen_id }`; the
home replays. **Not** a durable queue — bounded only.

## Dev workflow

Run the daemon and a unit-test sweep:

```sh
cargo run -p brainwires-home -- --bind 127.0.0.1:7878
cargo test -p brainwires-home
```

The M1 unit test in `src/webrtc.rs` spins up two in-process WebRTC peers,
runs the offer/answer dance manually, opens the canonical `"a2a"` data
channel, and round-trips a ping/pong frame. Passing this test is the
gate to wiring the same peer into the axum signaling routes in M3.

For end-to-end PWA → home dev (M5+): point `web/src/home-signaling.js` at
`http://127.0.0.1:7878` and flip the dev toggle in the PWA Settings panel.

## Pointing the PWA at the daemon

The PWA and the home daemon always live on different origins — PWA in a
browser, daemon on `127.0.0.1:7878` (behind a tunnel for remote access).
That's a cross-origin request, so the daemon ships with a configurable
CORS layer. Two flows:

### 1. Localhost dev

PWA served from `localhost:8080` (the chat-PWA's `docker compose` default;
see `extras/brainwires-chat-pwa/docker-compose.yml` `HOST_PORT`), daemon on
`127.0.0.1:7878`.

```sh
# daemon — defaults already allow http://localhost:8080
cargo run -p brainwires-home -- --bind 127.0.0.1:7878

# or explicit (equivalent for localhost dev):
cargo run -p brainwires-home -- --cors-origin http://localhost:8080
```

PWA: open `http://localhost:8080/?home=http://127.0.0.1:7878` (or set the
home URL in the Settings panel).

### 2. Tunneled production

PWA at e.g. `https://chat.example.com`, daemon at `127.0.0.1:7878` behind
a tunnel reachable as `https://home.example.com`.

```sh
cargo run -p brainwires-home -- \
    --bind 127.0.0.1:7878 \
    --cors-origin https://chat.example.com
```

PWA: `https://chat.example.com/?home=https://home.example.com`.

Adding even one `--cors-origin` clears the dev defaults — production
daemons should not silently accept `http://localhost:8080` next to their
real origin.

### CORS

The home daemon attaches a `tower_http::cors::CorsLayer` to every route.
Three modes:

| Flag                                                    | Behaviour |
|---------------------------------------------------------|-----------|
| (none)                                                  | Allow chat-PWA dev origins: `http://localhost:8080`, `http://127.0.0.1:8080`, `http://localhost:5173`, `http://127.0.0.1:5173`. |
| `--cors-origin <URL>` (repeatable)                      | Exact-match allow-list. First call clears the dev defaults. |
| `--cors-permissive` (env `BRAINWIRES_HOME_CORS_PERMISSIVE`) | Allow any origin (`Access-Control-Allow-Origin: *`). **Dev only.** |

Methods allowed: `GET POST DELETE OPTIONS`. Headers allowed:
`content-type`, `authorization`, `cf-access-client-id`,
`cf-access-client-secret` (the latter two ride along today even though M5
doesn't use them — they're the M8 pairing flow's auth headers).
`Access-Control-Max-Age: 600` (10 min) on preflight responses. **No**
credentials mode — the PWA carries Bearer tokens (M8), not cookies.

Why permissive is dev-only: any web page in any browser tab can preflight-
poke a permissive daemon and discover its endpoints. Production should
always pin to one origin via `--cors-origin`.

### TURN credentials

`POST /signal/session` returns an `ice_servers` array the PWA hands to
`new RTCPeerConnection({ iceServers: ... })`. ICE then walks the list:
direct → STUN-discovered server-reflexive → TURN-relayed.

Why TURN matters: roughly 10–15% of mobile carriers deploy symmetric NAT,
where each outbound destination gets its own external port mapping. STUN
alone cannot punch through symmetric NAT — the PWA's port observed by
the STUN server is not the port the home daemon's traffic would hit. A
TURN relay solves this by giving both ends a stable rendezvous server.

The home daemon mints a fresh, short-lived (~10 min) Cloudflare Calls
TURN credential per session. **The PWA never holds the Calls API
token** — it only sees the resulting `username`/`credential` pair, scoped
to one session. Token rotation is a pure home-side operation.

| Flag                                              | Env             | Default | Description |
|---------------------------------------------------|-----------------|---------|-------------|
| `--cf-turn-key-id <ID>`                           | `CF_TURN_KEY_ID`| —       | Cloudflare Calls TURN key id |
| `--cf-turn-token <TOKEN>`                         | `CF_TURN_TOKEN` | —       | Cloudflare **Calls** API token (NOT a Tunnel token) |
| `--turn-ttl <SECONDS>`                            | `TURN_TTL`      | `600`   | Credential lifetime; floored at 60 s |

Both `--cf-turn-key-id` and `--cf-turn-token` must be set together. If
only one is supplied, the daemon logs a warning and falls back to STUN-
only — silently dropping a half-configured TURN would mask an obvious
typo.

Get a key:

1. `dashboard.cloudflare.com → Calls → TURN keys → Create TURN key`.
2. Copy the key id (numeric) and the API token. The token's only scope
   is the Calls API; it cannot reach Tunnel, R2, or Workers.
3. Pass them to the daemon:

   ```sh
   cargo run -p brainwires-home -- \
       --bind 127.0.0.1:7878 \
       --cors-origin https://chat.example.com \
       --cf-turn-key-id 1a2b3c4d... \
       --cf-turn-token cf-calls-token-abc...
   ```

**Default behaviour without TURN**: the daemon returns just the two
public STUN servers. That's enough for ~85% of home networks, but expect
"can't connect over LTE/5G" reports until TURN is wired up. The
verification path is manual — load the PWA on a phone over cellular,
issue a `ping`, watch for a `pong` reply over the data channel.

**Failure mode**: if the Cloudflare API itself is unreachable mid-
session (timeout / 5xx / DNS), the daemon logs a warning and returns
STUN-only for that session rather than failing the request. Connection
establishment must not hard-depend on the TURN API being up — most
sessions don't actually need TURN to complete.

### Bind address

Default `127.0.0.1:7878` — loopback only. The tunnel client (cloudflared,
ngrok, etc.) connects to this loopback address and is what's exposed to
the public internet:

```sh
cloudflared tunnel run --url http://localhost:7878 brainwires-home
```

Don't bind `0.0.0.0` unless you've thought through the firewall and
authentication implications. The whole point of the tunnel is that
public exposure is a separate concern from the daemon's listener.

### M6 verification

Manual round-trip (run on the same hardware that hosts the tunnel):

1. Start the daemon with the right CORS allow-list for your PWA origin.
2. Open the PWA in a browser. Set the home URL to the tunnel hostname.
3. Issue a `ping` from the PWA dev console; expect a `pong`-shaped JSON-
   RPC reply over the WebRTC data channel.

`cargo test -p brainwires-home` covers preflight allow / disallow paths
under each mode in process — but the actual phone → tunnel → home
round-trip is a manual test the user runs against their existing tunnel.

**SRI**: not relevant to M6. The chat PWA vendors all assets at build
time (KaTeX, pdfjs, highlight.js, the WASM module — see
`extras/brainwires-chat-pwa/web/build.mjs`); there are no CDN scripts to
integrity-check. The service worker hashes its own static cache via the
`STATIC_ASSETS` table in `build.mjs`, which is a separate offline-cache
integrity property.

## Production

The home daemon expects a Cloudflare Tunnel (or any reverse tunnel that
lands on `127.0.0.1:7878`) to be running on the same host. The user
provides:

- `cloudflared` binary + tunnel credentials (one-time `cloudflared tunnel
  login` + `cloudflared tunnel create brainwires-home`),
- A hostname that maps to `http://127.0.0.1:7878` in the tunnel config,
- Cloudflare Access service-token pair, bound at pairing time, sent on
  every signaling request as `CF-Access-Client-Id` / `CF-Access-Client-Secret`,
- Optional Cloudflare Calls API token for TURN credential minting (M7).

The daemon also enforces its own `Authorization: Bearer <device_token>`
on every signaling request — defence in depth, so a leaked CF service
token alone cannot reach the agent.

## Phase 2 verification (manual)

These are the user-facing acceptance tests that closed out Phase 2. All
seven require real hardware (a paired phone or laptop talking to a home
machine through a Cloudflare Tunnel) so they're driven manually rather
than from `cargo test`. The unit suites in `cargo test -p brainwires-home`
and `cd web && npm test` cover the underlying primitives in process.

| ID | What it exercises                                                                          |
|----|--------------------------------------------------------------------------------------------|
| T1 | `brainwires-home pair` flow end-to-end — QR scan + 6-digit confirm, bundle persisted       |
| T2 | "Home agent" provider becomes selectable in the chat UI once paired                        |
| T3 | A2A `message/send` round-trip — text reply rendered in the chat bubble                     |
| T4 | Multimodal — image + PDF attachment uploaded via `bin/begin`+`bin/chunk`+`bin/end`         |
| T5 | Cellular path — TURN-relayed connection from phone over LTE through CF Calls TURN          |
| T6 | Reconnect — drop wifi mid-conversation; ICE restart recovers the session within ~10 s      |
| T7 | Unpair — "Remove paired device" disconnects + clears the bundle + hides the provider       |

## Known limitations

These are deferred follow-ups, not regressions — Phase 2's surface is
intentionally small. Each is independently shippable as a future
milestone.

- **Streaming via `message/stream`.** The home provider does one
  `message/send` round-trip per turn. True incremental streaming needs
  daemon-side `Provider::stream_chat` plumbing plus a PWA-side
  `home-stream.js` adapter that subscribes to JSON-RPC notifications.
  The chat UI's `chat_chunk` plumbing already supports incremental
  rendering, so the change is local to the provider adapter.

- **Length-prefixed binary framing.** M11 uplinks file payloads as
  base64-in-text frames (`bin/chunk { data: "<b64>" }`). A length-
  prefixed binary framing on the same data channel (`[u32 LE
  length][bytes]`) would cut ~33% of bandwidth and CPU. Not user-
  visible until somebody pushes >100 MB through one upload.

- **CF Access service-token minting from the daemon.** Today the
  operator pre-provisions one `(client_id, client_secret)` pair via
  `--cf-access-client-id` / `--cf-access-client-secret`. A future
  improvement is for the daemon to mint a fresh service-token pair per
  paired device against the Cloudflare API — same security gain CF
  Calls TURN already has (M7), applied to the signaling path.

## What this crate is not

- **Not** a relay or mesh node. One PWA ↔ one home daemon.
- **Not** a STUN/TURN server. The daemon mints CF Calls credentials and
  hands them to the PWA; it does not proxy media.
- **Not** a multi-tenant gateway. For that, see `brainclaw-gateway`. This
  daemon serves a single user's PWA(s).
- **Not** a durable agent host. Agents run in-process; if the daemon
  restarts the conversation reconnects (see Reconnect above), but in-flight
  requests beyond the bounded ring buffer are dropped.

## License

MIT OR Apache-2.0 — same as the workspace.
