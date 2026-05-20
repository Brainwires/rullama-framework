/**
 * @module @brainwires/session
 *
 * Pluggable session-persistence for the Brainwires Agent Framework.
 *
 * The {@link SessionStore} interface is the single extension point;
 * {@link InMemorySessionStore} is the default for tests and ephemeral
 * sessions, and {@link DenoKvSessionStore} provides disk-backed
 * persistence via Deno's built-in KV store.
 *
 * Equivalent to Rust's `brainwires-session` crate.
 */

export { SessionError } from "./error.ts";
export {
  defaultListOptions,
  type ListOptions,
  SessionId,
  type SessionRecord,
} from "./types.ts";
export { defaultListPaginated, type SessionStore } from "./store.ts";
export { InMemorySessionStore } from "./memory.ts";
export { DenoKvSessionStore } from "./deno_kv.ts";
