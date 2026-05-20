# @brainwires/session

Pluggable session persistence for the Brainwires Agent Framework.

## What it does

Stores and retrieves agent conversation transcripts (`Message[]`) keyed by
an opaque `SessionId`. Backends are interchangeable — swap them out without
touching the rest of your code.

## Backends

- **`InMemorySessionStore`** — in-process Map, nothing persists across
  restarts. Use for tests and ephemeral sessions.
- **`DenoKvSessionStore`** — Deno KV-backed, persists to disk when opened
  against a file path, in-memory when opened with `":memory:"`. Replaces the
  Rust crate's SQLite backend with an idiomatic Deno-native option.

Want a different backend (Postgres, Redis, filesystem-JSON)? Implement the
`SessionStore` interface directly — there are five async methods.

## Example

```ts
import { Message } from "@brainwires/core";
import {
  DenoKvSessionStore,
  InMemorySessionStore,
  SessionId,
} from "@brainwires/session";

// In-memory — great for tests.
const mem = new InMemorySessionStore();
await mem.save(new SessionId("alice"), [Message.user("hi")]);

// Disk-backed — survives restarts.
const kv = await Deno.openKv("./sessions.kv");
const store = new DenoKvSessionStore(kv);
await store.save(new SessionId("bob"), [Message.user("ping")]);
```

## API

| method | purpose |
|---|---|
| `load(id)` | Return the transcript, or `null` for unknown sessions. |
| `save(id, messages)` | Atomic overwrite of the transcript. |
| `list()` | Metadata for every known session, `updated_at` ascending. |
| `listPaginated({ offset, limit })` | Same, with pagination. `limit: null` = unbounded. |
| `delete(id)` | Remove. Deleting an unknown id is a no-op. |

## Equivalent Rust crate

`brainwires-session` — same trait shape, same semantics. The SQLite backend
is replaced here by Deno KV to stay runtime-native.
