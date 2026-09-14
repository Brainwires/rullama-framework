/**
 * Shared value types used by every {@link SessionStore} implementation.
 *
 * Equivalent to Rust's `rullama_session::types` module.
 */

/** Opaque identifier for a persisted session. */
export class SessionId {
  /** The raw string identifier this SessionId wraps. */
  readonly value: string;

  /** Wrap a raw string as a SessionId. Equivalent to {@link SessionId.from}. */
  constructor(value: string) {
    this.value = value;
  }

  /** Build a SessionId from any string. */
  static from(s: string): SessionId {
    return new SessionId(s);
  }

  /** @deprecated Use {@link SessionId.from}; removed in 0.13. */
  // deno-lint-ignore no-misused-new
  static new(s: string): SessionId {
    return new SessionId(s);
  }

  /** Borrow the id as a plain string. */
  asStr(): string {
    return this.value;
  }

  /** Render the id as its underlying string (used by template literals and `String()`). */
  toString(): string {
    return this.value;
  }

  /**
   * Compare against another SessionId or a raw string by value.
   *
   * @param other Another SessionId, or a plain string id.
   * @returns `true` when both refer to the same underlying string.
   */
  equals(other: SessionId | string): boolean {
    return this.value === (other instanceof SessionId ? other.value : other);
  }
}

/** Metadata row returned by {@link SessionStore.list}. */
export interface SessionRecord {
  /** Identifier of the session this row describes. */
  id: SessionId;
  /** Number of messages in the transcript. */
  message_count: number;
  /** ISO 8601 timestamp — when the session was first persisted. */
  created_at: string;
  /** ISO 8601 timestamp — when the session was last written. */
  updated_at: string;
}

/**
 * Pagination window passed to {@link SessionStore.listPaginated}.
 *
 * `offset` rows are skipped; `limit` (when non-null) caps the returned count.
 * Defaults to `{ offset: 0, limit: null }`, equivalent to an unbounded `list`.
 */
export interface ListOptions {
  /** Number of leading rows to skip. */
  offset: number;
  /** Maximum number of rows to return, or `null` for no cap. */
  limit: number | null;
}

/** Default ListOptions — no offset, no limit. */
export function defaultListOptions(): ListOptions {
  return { offset: 0, limit: null };
}
