/**
 * In-memory {@link SessionStore} implementation.
 *
 * Intended for tests, ephemeral sessions, and embedding use-cases. Nothing
 * persists across process restarts.
 *
 * Equivalent to Rust's `InMemorySessionStore` in `brainwires_session`.
 */

import type { Message } from "@brainwires/core";
import type { SessionStore } from "./store.ts";
import { defaultListPaginated } from "./store.ts";
import type { ListOptions, SessionRecord } from "./types.ts";
import { SessionId } from "./types.ts";

interface Entry {
  messages: Message[];
  created_at: string;
  updated_at: string;
}

/** In-memory session store. */
export class InMemorySessionStore implements SessionStore {
  private readonly entries = new Map<string, Entry>();

  load(id: SessionId): Promise<Message[] | null> {
    const entry = this.entries.get(id.value);
    return Promise.resolve(entry ? [...entry.messages] : null);
  }

  save(id: SessionId, messages: Message[]): Promise<void> {
    const now = new Date().toISOString();
    const existing = this.entries.get(id.value);
    if (existing) {
      existing.messages = [...messages];
      existing.updated_at = now;
    } else {
      this.entries.set(id.value, {
        messages: [...messages],
        created_at: now,
        updated_at: now,
      });
    }
    return Promise.resolve();
  }

  list(): Promise<SessionRecord[]> {
    const out: SessionRecord[] = [];
    for (const [idStr, e] of this.entries) {
      out.push({
        id: new SessionId(idStr),
        message_count: e.messages.length,
        created_at: e.created_at,
        updated_at: e.updated_at,
      });
    }
    out.sort((a, b) => a.updated_at.localeCompare(b.updated_at));
    return Promise.resolve(out);
  }

  listPaginated(opts: ListOptions): Promise<SessionRecord[]> {
    return defaultListPaginated(this, opts);
  }

  delete(id: SessionId): Promise<void> {
    this.entries.delete(id.value);
    return Promise.resolve();
  }
}
