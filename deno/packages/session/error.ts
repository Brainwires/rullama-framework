/**
 * Typed errors surfaced by session stores.
 *
 * Equivalent to Rust's `brainwires_session::error` module.
 */

/** Errors surfaced by session-store implementations. */
export class SessionError extends Error {
  readonly kind: "serialization" | "storage";

  constructor(kind: "serialization" | "storage", message: string) {
    super(`session ${kind}: ${message}`);
    this.kind = kind;
    this.name = "SessionError";
  }

  static serialization(message: string): SessionError {
    return new SessionError("serialization", message);
  }

  static storage(message: string): SessionError {
    return new SessionError("storage", message);
  }
}
