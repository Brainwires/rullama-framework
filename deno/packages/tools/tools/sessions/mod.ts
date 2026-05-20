/**
 * Session-control tools exposed to the agent.
 *
 * The agent uses these to inspect or orchestrate *other* chat sessions
 * running in the same host process — listing peers, reading their history,
 * pushing a message into one, or spawning a fresh sub-session.
 *
 * Session state lives outside this package (in the host), so this module
 * only defines the tool schemas plus a {@link SessionBroker} interface that
 * the host implements over its actual registry.
 *
 * Equivalent to Rust's `brainwires_tools::sessions` module.
 */

export {
  defaultSpawnRequest,
  type SessionBroker,
  SessionId,
  type SessionMessage,
  type SessionSummary,
  type SpawnedSession,
  type SpawnRequest,
} from "./broker.ts";

export {
  CTX_METADATA_SESSION_ID,
  MAX_HISTORY_LIMIT,
  SessionsTool,
  TOOL_SESSIONS_HISTORY,
  TOOL_SESSIONS_LIST,
  TOOL_SESSIONS_SEND,
  TOOL_SESSIONS_SPAWN,
} from "./sessions_tool.ts";
