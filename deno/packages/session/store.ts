/**
 * {@link SessionStore} interface — the single extension point for pluggable
 * session persistence.
 *
 * Equivalent to Rust's `SessionStore` trait in `brainwires_session`.
 */

import type { Message } from "@brainwires/core";
import type { ListOptions, SessionId, SessionRecord } from "./types.ts";

/**
 * Interface implemented by every session-persistence backend.
 *
 * Implementations must be safe to call concurrently from any async context.
 */
export interface SessionStore {
  /**
   * Load a session's full transcript. Returns `null` when the id is not
   * known — callers should treat that as "fresh session".
   */
  load(id: SessionId): Promise<Message[] | null>;

  /**
   * Overwrite a session's full transcript. Implementations should persist
   * the provided array atomically — a crash mid-write must leave the store
   * with either the old or new transcript, never a partial one.
   */
  save(id: SessionId, messages: Message[]): Promise<void>;

  /**
   * Enumerate every session the store knows about, sorted by `updated_at`
   * ascending. Returns metadata only — use {@link load} to read content.
   */
  list(): Promise<SessionRecord[]>;

  /**
   * Enumerate sessions with offset/limit pagination. Default implementations
   * slice in memory; backends that can push the window down to storage
   * should override this.
   */
  listPaginated(opts: ListOptions): Promise<SessionRecord[]>;

  /** Remove a session. Deleting an unknown id is a no-op, not an error. */
  delete(id: SessionId): Promise<void>;
}

/**
 * Default implementation of {@link SessionStore.listPaginated} built on
 * {@link SessionStore.list}. Backends can call this to get a correct
 * memory-slice fallback, or override for a storage-native window.
 */
export async function defaultListPaginated(
  store: Pick<SessionStore, "list">,
  opts: ListOptions,
): Promise<SessionRecord[]> {
  const all = await store.list();
  const start = Math.min(opts.offset, all.length);
  const end = opts.limit === null
    ? all.length
    : Math.min(start + opts.limit, all.length);
  return all.slice(start, end);
}
