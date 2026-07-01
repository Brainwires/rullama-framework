/**
 * @module @rullama/tool-builtins
 *
 * Built-in tool implementations: BashTool, FileOpsTool, GitTool, WebTool,
 * SearchTool, SemanticSearchTool, CalendarTool, SessionsTool.
 *
 * Extracted from `@rullama/tools` in v0.11.0 to mirror Rust's
 * `rullama-tool-builtins`. The executor / registry / sanitizer
 * framework lives in `@rullama/tool-runtime`.
 */

export { BashTool } from "./bash.ts";
export type { OutputMode, StderrMode } from "./bash.ts";

export { FileOpsTool } from "./file_ops.ts";

export { GitTool } from "./git.ts";

export { SearchTool } from "./search.ts";

export { WebTool } from "./web.ts";

// Semantic code search (RAG-backed)
export { SemanticSearchTool } from "./semantic_search.ts";

// Calendar (Google + CalDAV)
export {
  type Attendee,
  type AttendeeStatus,
  type BusyStatus,
  CalDavClient,
  type CalendarConfig,
  type CalendarEvent,
  type CalendarInfo,
  type CalendarProvider,
  CalendarTool,
  type FreeBusySlot,
  GoogleCalendarClient,
  newAttendee,
  type Recurrence,
  type RecurrenceFreq,
} from "./calendar/mod.ts";

// Sessions (list / history / send / spawn)
export {
  CTX_METADATA_SESSION_ID,
  defaultSpawnRequest,
  MAX_HISTORY_LIMIT,
  type SessionBroker,
  SessionId,
  type SessionMessage,
  type SessionsTool,
  SessionsTool as SessionsToolClass,
  type SessionSummary,
  type SpawnedSession,
  type SpawnRequest,
  TOOL_SESSIONS_HISTORY,
  TOOL_SESSIONS_LIST,
  TOOL_SESSIONS_SEND,
  TOOL_SESSIONS_SPAWN,
} from "./sessions/mod.ts";
