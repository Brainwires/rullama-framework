/**
 * Built-in tool implementations re-exports.
 */

export { BashTool } from "./bash.ts";
export type { OutputMode, StderrMode } from "./bash.ts";

export { FileOpsTool } from "./file_ops.ts";

export { GitTool } from "./git.ts";

export { SearchTool } from "./search.ts";

export { ValidationTool } from "./validation.ts";
export { isExportLine, extractExportName } from "./validation.ts";

export { WebTool } from "./web.ts";

export {
  executeOpenApiTool,
  executeOpenApiToolWithEndpoint,
  openApiToToolDefs,
  openApiToTools,
} from "./openapi.ts";
export type {
  HttpMethod,
  OpenApiEndpoint,
  OpenApiParam,
  OpenApiToolDef,
} from "./openapi.ts";

// OAuth (PKCE, client-credentials, token refresh, pluggable token store)
export {
  authorizationCodePkceConfig,
  base64UrlEncode,
  clientCredentialsConfig,
  InMemoryTokenStore,
  isTokenExpired,
  newPkceChallenge,
  OAuthClient,
  type OAuthConfig,
  type OAuthFlow,
  type OAuthToken,
  type OAuthTokenStore,
  type PkceChallenge,
  pkceAuthorizationUrl,
} from "./oauth.ts";

// Calendar (Google + CalDAV)
export {
  CalDavClient,
  CalendarTool,
  type CalendarConfig,
  type CalendarProvider,
  GoogleCalendarClient,
  newAttendee,
  type Attendee,
  type AttendeeStatus,
  type BusyStatus,
  type CalendarEvent,
  type CalendarInfo,
  type FreeBusySlot,
  type Recurrence,
  type RecurrenceFreq,
} from "./calendar/mod.ts";

// Tool search + embedding
export {
  DEFAULT_SEARCH_MODE,
  type SearchMode,
  ToolSearchTool,
} from "./tool_search.ts";
export {
  cosineSimilarity,
  ToolEmbeddingIndex,
} from "./tool_embedding.ts";

// Semantic code search (RAG-backed)
export { SemanticSearchTool } from "./semantic_search.ts";

// Sessions (list / history / send / spawn)
export {
  CTX_METADATA_SESSION_ID,
  defaultSpawnRequest,
  MAX_HISTORY_LIMIT,
  type SessionBroker,
  SessionId,
  type SessionMessage,
  type SessionSummary,
  SessionsTool,
  type SpawnedSession,
  type SpawnRequest,
  TOOL_SESSIONS_HISTORY,
  TOOL_SESSIONS_LIST,
  TOOL_SESSIONS_SEND,
  TOOL_SESSIONS_SPAWN,
} from "./sessions/mod.ts";
