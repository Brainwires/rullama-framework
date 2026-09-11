/**
 * Typed errors surfaced by session stores.
 *
 * Equivalent to Rust's `rullama_session::error` module.
 */

/** Errors surfaced by session-store implementations. */
export class SessionError extends Error {
  /**
   * Failure category: `"serialization"` when a transcript could not be
   * encoded/decoded, `"storage"` when the backend itself failed.
   */
  readonly kind: "serialization" | "storage";

  /**
   * Create a SessionError; the final message is prefixed with
   * `session <kind>: `.
   *
   * @param kind Failure category.
   * @param message Backend-specific detail.
   */
  constructor(kind: "serialization" | "storage", message: string) {
    super(`session ${kind}: ${message}`);
    this.kind = kind;
    this.name = "SessionError";
  }

  /** Build a `"serialization"` error (transcript could not be encoded or decoded). */
  static serialization(message: string): SessionError {
    return new SessionError("serialization", message);
  }

  /** Build a `"storage"` error (the backing store failed to read or write). */
  static storage(message: string): SessionError {
    return new SessionError("storage", message);
  }
}
