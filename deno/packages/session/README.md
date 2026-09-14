# @rullama/session

Pluggable session persistence for the rullama.

## What it does

Stores and retrieves agent conversation transcripts (`Message[]`) keyed by an
opaque `SessionId`. Backends are interchangeable — swap them out without
touching the rest of your code.

## Install

```sh
deno add jsr:@rullama/session
```

## Backends

- **`InMemorySessionStore`** — in-process `Map`, nothing persists across
  restarts. No permissions or unstable features needed; use it for tests,
  ephemeral sessions, and embedding.
- **`DenoKvSessionStore`** — Deno KV-backed, persists to disk when opened
  against a file path, in-memory when opened with `":memory:"`. Replaces the
  Rust crate's SQLite backend with an idiomatic Deno-native option. Deno KV is
  an **unstable** API: enable the `kv` feature with `"unstable": ["kv"]` in your
  `deno.json` (or run with `--unstable-kv`). The store wraps an already-open
  `Deno.Kv` and never closes it — you own the handle's lifetime.

Want a different backend (Postgres, Redis, filesystem-JSON)? Implement the
`SessionStore` interface directly — there are five async methods.

## Example

```ts
import { Message } from "@rullama/core";
import {
  DenoKvSessionStore,
  InMemorySessionStore,
  SessionId,
} from "@rullama/session";

// In-memory — great for tests.
const mem = new InMemorySessionStore();
await mem.save(SessionId.from("alice"), [Message.user("hi")]);
const transcript = await mem.load(SessionId.from("alice")); // Message[] | null

// Disk-backed — survives restarts (needs the `kv` unstable feature).
const kv = await Deno.openKv("./sessions.kv");
const store = new DenoKvSessionStore(kv);
await store.save(SessionId.from("bob"), [Message.user("ping")]);
for (const rec of await store.listPaginated({ offset: 0, limit: 10 })) {
  console.log(rec.id.asStr(), rec.message_count, rec.updated_at);
}
kv.close();
```

Every backend throws `SessionError` (with `kind` of `"serialization"` or
`"storage"`) for failures it can classify.

## API

| method                             | purpose                                                   |
| ---------------------------------- | --------------------------------------------------------- |
| `load(id)`                         | Return the transcript, or `null` for unknown sessions.    |
| `save(id, messages)`               | Atomic overwrite of the transcript.                       |
| `list()`                           | Metadata for every known session, `updated_at` ascending. |
| `listPaginated({ offset, limit })` | Same, with pagination. `limit: null` = unbounded.         |
| `delete(id)`                       | Remove. Deleting an unknown id is a no-op.                |

## Equivalent Rust crate

`rullama-session` — same trait shape, same semantics. The SQLite backend is
replaced here by Deno KV to stay runtime-native.
