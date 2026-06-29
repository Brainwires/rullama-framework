# brainwires-memory-server

A Mem0-compatible memory REST API server for Brainwires agents. (Renamed
from `brainwires-memory-service` in v0.10.x — the lib crate
`brainwires-memory` now houses the framework's internal tiered memory
primitives, so this server got the disambiguating `-server` suffix.)

Gives every agent a persistent, per-user memory store accessible over HTTP — point any Mem0 SDK client (or plain `curl`) at it and your agents remember things between sessions.

Storage is delegated to [`brainwires-knowledge`](../../crates/brainwires-knowledge) (LanceDB-backed thought store with semantic search). Tenant isolation is enforced on every request via `user_id`, which is passed through to the knowledge layer as its `owner_id` scope — a request for user A can never see, update, or delete data belonging to user B.

As Nate B Jones put it: *"whoever solves orchestration at infrastructure grade is going to own the most valuable position in the agent stack."* Memory is how agents build context across sessions; this service is the persistence layer for that.

## Quick start

```sh
# Run with default settings (localhost:8765, ~/.local/share/brainwires/memory/)
cargo run --bin brainwires-memory-server

# Override via environment variables
MEMORY_HOST=0.0.0.0 MEMORY_PORT=8765 MEMORY_DB=/data/memory \
  cargo run --bin brainwires-memory-server
```

## Tenant isolation (`user_id` is required)

Every endpoint that touches stored memories requires a `user_id` — either in the request body (for `POST`/`PATCH`) or as a query parameter (for `GET`/`DELETE`). Requests missing `user_id` receive HTTP 400.

The `user_id` is forwarded to the underlying knowledge layer as `owner_id`, which guarantees:

- A `GET`/`POST search` request for user A never returns user B's memories.
- A `PATCH`/`DELETE` request for user A cannot mutate user B's memories; cross-tenant attempts return 404 (identical to a missing ID, so the existence of other tenants' memories is not leaked).

## Auth

This service does not ship with bearer-token auth. If you need authentication, layer it via a reverse proxy (Traefik, Caddy, nginx, etc.) that validates credentials before forwarding to `brainwires-memory`.

## API

### Add memory

```http
POST /v1/memories
Content-Type: application/json

{
  "memory": "The user prefers concise answers.",
  "user_id": "user-42"
}
```

Or pass raw message history (role + content pairs) and the service extracts each turn as a separate memory:

```http
POST /v1/memories
{
  "messages": [
    { "role": "user", "content": "I prefer Python over Ruby." },
    { "role": "assistant", "content": "Got it, I'll use Python for examples." }
  ],
  "user_id": "user-42"
}
```

Response:
```json
{
  "results": [
    { "id": "uuid", "memory": "I prefer Python over Ruby.", "event": "add" }
  ]
}
```

### List memories

```http
GET /v1/memories?user_id=user-42&page=1&page_size=20
```

### Get a memory

```http
GET /v1/memories/{id}?user_id=user-42
```

### Search memories

```http
POST /v1/memories/search
{
  "query": "preferred programming language",
  "user_id": "user-42",
  "limit": 5
}
```

Search is a semantic (vector) query against the knowledge layer's embedding index.

### Update a memory

```http
PATCH /v1/memories/{id}
{ "memory": "Updated content.", "user_id": "user-42" }
```

### Delete a memory

```http
DELETE /v1/memories/{id}?user_id=user-42
```

### Delete all memories for a user

```http
DELETE /v1/memories?user_id=user-42
```

### Health check

```http
GET /health
→ { "status": "ok" }
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_HOST` | `127.0.0.1` | Bind address |
| `MEMORY_PORT` | `8765` | Listen port |
| `MEMORY_DB` | `~/.local/share/brainwires/memory` | Knowledge storage directory (contains `brain.lance/`, `pks.db`, `bks.db`) |
| `RUST_LOG` | `brainwires_memory_service=info` | Log filter |

## Using with Mem0 SDK

```python
from mem0 import MemoryClient

client = MemoryClient(host="http://localhost:8765", api_key="unused")
client.add("I prefer Rust over Go", user_id="user-42")
results = client.search("programming language preference", user_id="user-42")
```

## Architecture

```
┌──────────────────────────────────┐
│        brainwires-memory         │
│                                  │
│  POST /v1/memories               │
│  GET  /v1/memories               │  Axum HTTP server
│  POST /v1/memories/search   ─────┼──► BrainClient (brainwires-knowledge)
│  PATCH/DELETE /v1/memories/{id}  │    └─► LanceDB (vectors)
│  GET  /health                    │    └─► SQLite (PKS/BKS)
└──────────────────────────────────┘
```

The server is a thin adapter over `brainwires-knowledge` — all storage decisions (embedding model, vector index, backend selection) are made by the knowledge crate. Tenant isolation is implemented there, via the `owner_id` column on every thought row.

## License

MIT OR Apache-2.0
