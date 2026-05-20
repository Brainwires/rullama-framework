/**
 * Types and interface for the host-provided session registry.
 *
 * `@brainwires/tools` is a framework package — it does not know about the
 * gateway's per-user session map or the concrete agent machinery. The
 * {@link SessionBroker} interface bridges that gap: the host implements it
 * against its real registry and hands an instance to {@link SessionsTool}.
 *
 * Equivalent to Rust's `brainwires_tools::sessions::broker` module.
 */

/** Opaque identifier for a chat session. */
export class SessionId {
  readonly value: string;

  constructor(value: string) {
    this.value = value;
  }

  /** Borrow the underlying id as a string slice. */
  asStr(): string {
    return this.value;
  }

  toString(): string {
    return this.value;
  }

  /** Structural equality with another SessionId or raw string. */
  equals(other: SessionId | string): boolean {
    return this.value === (other instanceof SessionId ? other.value : other);
  }
}

/** Summary metadata for a single session, returned by `sessions_list`. */
export interface SessionSummary {
  id: SessionId;
  /** Originating channel (e.g. "discord", "web", "internal"). */
  channel: string;
  /** Peer handle — user id on the channel, or "spawned-by-<parent>". */
  peer: string;
  /** When the session was first created (ISO 8601). */
  created_at: string;
  /** When the session last received or produced a message (ISO 8601). */
  last_active: string;
  /** Number of messages currently in the session's transcript. */
  message_count: number;
  /** Parent session that spawned this one, if any. */
  parent: SessionId | null;
}

/** A single message from a session's transcript. */
export interface SessionMessage {
  /** "user" | "assistant" | "system" | "tool". */
  role: string;
  /** Message text. Tool calls/results are stringified. */
  content: string;
  /** When the message was recorded (ISO 8601). */
  timestamp: string;
}

/** Parameters for {@link SessionBroker.spawn}. */
export interface SpawnRequest {
  /** Initial user message to seed the new session with. */
  prompt: string;
  /** Optional provider/model override. null = inherit from parent. */
  model: string | null;
  /** Optional system prompt override. null = inherit. */
  system: string | null;
  /** Tools to allow in the spawned session. null = inherit parent's toolset. */
  tools: string[] | null;
  /**
   * If true, block until the spawned session produces its first assistant
   * message (or {@link wait_timeout_secs} elapses) and return that in the
   * tool result. Default: false.
   */
  wait_for_first_reply: boolean;
  /** Seconds to wait when wait_for_first_reply is true. Default: 60. */
  wait_timeout_secs: number;
}

/** Default factory for a blank SpawnRequest. */
export function defaultSpawnRequest(): SpawnRequest {
  return {
    prompt: "",
    model: null,
    system: null,
    tools: null,
    wait_for_first_reply: false,
    wait_timeout_secs: 60,
  };
}

/** Result of {@link SessionBroker.spawn}. */
export interface SpawnedSession {
  id: SessionId;
  /** Set iff `wait_for_first_reply` was true and the reply arrived in time. */
  first_reply: SessionMessage | null;
}

/**
 * Host-provided bridge from session tools to the real session registry.
 */
export interface SessionBroker {
  /** List every live session the host knows about. */
  list(): Promise<SessionSummary[]>;
  /**
   * Read a session's transcript, newest-last, capped at `limit` entries
   * (null = use the host's sensible default).
   */
  history(
    id: SessionId,
    limit: number | null,
  ): Promise<SessionMessage[]>;
  /**
   * Inject a user-role message into `id`'s inbound queue. Fire-and-forget —
   * the target session processes it asynchronously.
   */
  send(id: SessionId, text: string): Promise<void>;
  /**
   * Create a new session as a child of `parent`, seeded with `req.prompt`.
   */
  spawn(parent: SessionId, req: SpawnRequest): Promise<SpawnedSession>;
}
