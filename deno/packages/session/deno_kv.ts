/**
 * Deno KV-backed {@link SessionStore} implementation.
 *
 * Deno KV is a key-value store built into the Deno runtime — it replaces
 * the Rust crate's `SqliteSessionStore` with an idiomatic Deno-native
 * backend. Same atomicity guarantees; no additional dependencies.
 *
 * To persist across restarts, pass a filesystem path when opening Deno.Kv:
 *
 * ```ts
 * const kv = await Deno.openKv("./sessions.kv");
 * const store = new DenoKvSessionStore(kv);
 * ```
 *
 * For tests, open the default (in-memory) instance:
 *
 * ```ts
 * const kv = await Deno.openKv(":memory:");
 * const store = new DenoKvSessionStore(kv);
 * ```
 */

import type { MessageData } from "@brainwires/core";
import { Message } from "@brainwires/core";
import type { SessionStore } from "./store.ts";
import { defaultListPaginated } from "./store.ts";
import type { ListOptions, SessionRecord } from "./types.ts";
import { SessionId } from "./types.ts";

/**
 * Persisted session row stored in KV. Messages are kept as plain JSON so
 * that a future runtime-native encoder can swap in without a migration.
 */
interface KvRow {
  messages: MessageData[];
  message_count: number;
  created_at: string;
  updated_at: string;
}

const KEY_PREFIX = ["brainwires", "session"] as const;

/** Session store backed by Deno KV. */
export class DenoKvSessionStore implements SessionStore {
  readonly kv: Deno.Kv;

  constructor(kv: Deno.Kv) {
    this.kv = kv;
  }

  private key(id: SessionId): Deno.KvKey {
    return [...KEY_PREFIX, id.value];
  }

  async load(id: SessionId): Promise<Message[] | null> {
    const row = await this.kv.get<KvRow>(this.key(id));
    if (row.value === null) return null;
    return row.value.messages.map((m) => new Message(m));
  }

  async save(id: SessionId, messages: Message[]): Promise<void> {
    const now = new Date().toISOString();
    const existing = await this.kv.get<KvRow>(this.key(id));
    const created_at = existing.value?.created_at ?? now;
    // Message implements MessageData, so a plain copy of role/content/… is
    // round-trippable through Deno KV's structured-clone serializer.
    const payload: KvRow = {
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
        name: m.name,
        metadata: m.metadata,
      })),
      message_count: messages.length,
      created_at,
      updated_at: now,
    };
    await this.kv.set(this.key(id), payload);
  }

  async list(): Promise<SessionRecord[]> {
    const out: SessionRecord[] = [];
    for await (const entry of this.kv.list<KvRow>({ prefix: [...KEY_PREFIX] })) {
      const idStr = String(entry.key[entry.key.length - 1]);
      out.push({
        id: new SessionId(idStr),
        message_count: entry.value.message_count,
        created_at: entry.value.created_at,
        updated_at: entry.value.updated_at,
      });
    }
    out.sort((a, b) => a.updated_at.localeCompare(b.updated_at));
    return out;
  }

  listPaginated(opts: ListOptions): Promise<SessionRecord[]> {
    return defaultListPaginated(this, opts);
  }

  async delete(id: SessionId): Promise<void> {
    await this.kv.delete(this.key(id));
  }
}
