/**
 * @module types
 *
 * MCP Protocol Types and JSON-RPC 2.0 types.
 * Equivalent to Rust's `rullama-mcp/src/types.rs`.
 *
 * All MCP types are defined natively as JSON-compatible interfaces
 * matching the MCP specification — no external SDK dependency.
 */

// =============================================================================
// JSON-RPC 2.0 TYPES
// =============================================================================

/**
 * JSON-RPC 2.0 Request.
 * Equivalent to Rust `JsonRpcRequest`.
 */
export interface JsonRpcRequest {
  /** JSON-RPC version (always "2.0"). */
  jsonrpc: "2.0";
  /** Request identifier. */
  id: JsonRpcId;
  /** Method name. */
  method: string;
  /** Optional parameters. */
  params?: unknown;
}

/**
 * JSON-RPC 2.0 Response.
 * Equivalent to Rust `JsonRpcResponse`.
 */
export interface JsonRpcResponse {
  /** JSON-RPC version (always "2.0"). */
  jsonrpc: "2.0";
  /** Response identifier matching the request. */
  id: JsonRpcId;
  /** Result value on success. */
  result?: unknown;
  /** Error object on failure. */
  error?: JsonRpcError;
}

/**
 * JSON-RPC 2.0 Error.
 * Equivalent to Rust `JsonRpcError`.
 */
export interface JsonRpcError {
  /** Error code. */
  code: number;
  /** Error message. */
  message: string;
  /** Optional additional data. */
  data?: unknown;
}

/**
 * JSON-RPC 2.0 Notification (no id field).
 * Equivalent to Rust `JsonRpcNotification`.
 */
export interface JsonRpcNotification {
  /** JSON-RPC version (always "2.0"). */
  jsonrpc: "2.0";
  /** Notification method name. */
  method: string;
  /** Optional parameters. */
  params?: unknown;
}

/** JSON-RPC identifier — number, string, or null. */
export type JsonRpcId = number | string | null;

/**
 * Generic JSON-RPC message that could be a response or notification.
 * Used for bidirectional MCP communication where servers can send notifications.
 * Equivalent to Rust `JsonRpcMessage` enum.
 */
export type JsonRpcMessage =
  | { type: "response"; response: JsonRpcResponse }
  | { type: "notification"; notification: JsonRpcNotification };

/**
 * Create a new JSON-RPC request.
 * Equivalent to Rust `JsonRpcRequest::new`.
 */
export function createJsonRpcRequest(
  id: JsonRpcId,
  method: string,
  params?: unknown,
): JsonRpcRequest {
  const req: JsonRpcRequest = { jsonrpc: "2.0", id, method };
  if (params !== undefined) {
    req.params = params;
  }
  return req;
}

/**
 * Create a new JSON-RPC notification (no id field).
 * Equivalent to Rust `JsonRpcNotification::new`.
 */
export function createJsonRpcNotification(
  method: string,
  params?: unknown,
): JsonRpcNotification {
  const notif: JsonRpcNotification = { jsonrpc: "2.0", method };
  if (params !== undefined) {
    notif.params = params;
  }
  return notif;
}

/**
 * Check if a message is a response (has non-null id).
 * Equivalent to Rust `JsonRpcMessage::is_response`.
 */
export function isJsonRpcResponse(
  msg: JsonRpcMessage,
): msg is { type: "response"; response: JsonRpcResponse } {
  return msg.type === "response";
}

/**
 * Check if a message is a notification.
 * Equivalent to Rust `JsonRpcMessage::is_notification`.
 */
export function isJsonRpcNotification(
  msg: JsonRpcMessage,
): msg is { type: "notification"; notification: JsonRpcNotification } {
  return msg.type === "notification";
}

/**
 * Parse a raw JSON line into a JsonRpcMessage, discriminating by the "id" field.
 * Equivalent to the logic in Rust `StdioTransport::receive_message`.
 */
export function parseJsonRpcMessage(raw: string): JsonRpcMessage {
  const value = JSON.parse(raw);
  const hasValidId = value.id !== undefined && value.id !== null;
  if (hasValidId) {
    return { type: "response", response: value as JsonRpcResponse };
  }
  return { type: "notification", notification: value as JsonRpcNotification };
}

// =============================================================================
// MCP PROGRESS NOTIFICATION TYPES
// =============================================================================

/**
 * Progress notification parameters from MCP server.
 * Equivalent to Rust `ProgressParams`.
 */
export interface ProgressParams {
  /** Token identifying which request this progress is for. */
  progressToken: string;
  /** Current progress value. */
  progress: number;
  /** Total expected value (for calculating percentage). */
  total?: number;
  /** Human-readable progress message. */
  message?: string;
}

/**
 * Parsed MCP notification types.
 * Equivalent to Rust `McpNotification` enum.
 */
export type McpNotification =
  | { type: "progress"; params: ProgressParams }
  | { type: "unknown"; method: string; params?: unknown };

/**
 * Parse a JsonRpcNotification into a typed McpNotification.
 * Equivalent to Rust `McpNotification::from_notification`.
 */
export function parseNotification(notif: JsonRpcNotification): McpNotification {
  if (notif.method === "notifications/progress" && notif.params) {
    const params = notif.params as Record<string, unknown>;
    if (
      typeof params.progressToken === "string" &&
      typeof params.progress === "number"
    ) {
      return {
        type: "progress",
        params: params as unknown as ProgressParams,
      };
    }
  }
  return { type: "unknown", method: notif.method, params: notif.params };
}

// =============================================================================
// MCP INITIALIZATION TYPES
// =============================================================================

/**
 * MCP Initialize Request Parameters.
 * Equivalent to Rust `InitializeParams`.
 */
export interface InitializeParams {
  /** Protocol version string. */
  protocolVersion: string;
  /** Client capabilities. */
  capabilities: ClientCapabilities;
  /** Client identification info. */
  clientInfo: ClientInfo;
}

/**
 * MCP client identification.
 * Equivalent to Rust `ClientInfo`.
 */
export interface ClientInfo {
  /** Client name. */
  name: string;
  /** Client version. */
  version: string;
}

/**
 * MCP Initialize Result.
 * Equivalent to Rust `InitializeResult`.
 */
export interface InitializeResult {
  /** Protocol version string. */
  protocolVersion: string;
  /** Server capabilities. */
  capabilities: ServerCapabilities;
  /** Server identification info. */
  serverInfo: ServerInfo;
}

/**
 * MCP server identification.
 * Equivalent to Rust `ServerInfo`.
 */
export interface ServerInfo {
  /** Server name. */
  name: string;
  /** Server version. */
  version: string;
}

// =============================================================================
// MCP CAPABILITY TYPES
// =============================================================================

/**
 * Client capabilities sent during initialization.
 * Equivalent to Rust `ClientCapabilities` (rmcp re-export).
 */
export interface ClientCapabilities {
  /** Roots capability. */
  roots?: Record<string, unknown>;
  /** Sampling capability. */
  sampling?: Record<string, unknown>;
  /** Experimental capabilities. */
  experimental?: Record<string, unknown>;
}

/**
 * Server capabilities returned during initialization.
 * Equivalent to Rust `ServerCapabilities` (rmcp re-export).
 */
export interface ServerCapabilities {
  /** Tools capability. */
  tools?: ToolsCapability;
  /** Resources capability. */
  resources?: ResourcesCapability;
  /** Prompts capability. */
  prompts?: PromptsCapability;
  /** Logging capability. */
  logging?: Record<string, unknown>;
  /** Experimental capabilities. */
  experimental?: Record<string, unknown>;
}

/**
 * Tools capability descriptor.
 * Equivalent to Rust `ToolsCapability`.
 */
export interface ToolsCapability {
  /** Whether the tool list may change. */
  listChanged?: boolean;
}

/**
 * Resources capability descriptor.
 * Equivalent to Rust `ResourcesCapability`.
 */
export interface ResourcesCapability {
  /** Whether the resource list may change. */
  listChanged?: boolean;
  /** Whether the server supports subscriptions. */
  subscribe?: boolean;
}

/**
 * Prompts capability descriptor.
 * Equivalent to Rust `PromptsCapability`.
 */
export interface PromptsCapability {
  /** Whether the prompt list may change. */
  listChanged?: boolean;
}

// =============================================================================
// MCP TOOL TYPES
// =============================================================================

/**
 * MCP Tool definition.
 * Equivalent to Rust `McpTool` (rmcp `Tool` re-export).
 */
export interface McpTool {
  /** Tool name. */
  name: string;
  /** Tool description. */
  description?: string;
  /** JSON Schema describing tool input. */
  inputSchema: Record<string, unknown>;
}

/**
 * Call tool request parameters.
 * Equivalent to Rust `CallToolParams` / `CallToolRequestParams`.
 */
export interface CallToolParams {
  /** Tool name. */
  name: string;
  /** Tool arguments as a JSON object. */
  arguments?: Record<string, unknown>;
}

/**
 * Call tool result.
 * Equivalent to Rust `CallToolResult`.
 */
export interface CallToolResult {
  /** Result content blocks. */
  content: Content[];
  /** Whether the tool call resulted in an error. */
  isError?: boolean;
}

/**
 * Content block within a tool result.
 * Equivalent to Rust `Content` (rmcp re-export).
 */
export type Content =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
  | { type: "resource"; resource: McpResource };

/**
 * Content type within a tool result (tagged union variant).
 * Equivalent to Rust `ToolResultContent`.
 */
export type ToolResultContent = Content;

/**
 * Tools List Response.
 * Equivalent to Rust `ListToolsResult`.
 */
export interface ListToolsResult {
  /** List of available tools. */
  tools: McpTool[];
}

// =============================================================================
// MCP RESOURCE TYPES
// =============================================================================

/**
 * MCP Resource definition.
 * Equivalent to Rust `McpResource` (rmcp `Resource` re-export).
 */
export interface McpResource {
  /** Resource URI. */
  uri: string;
  /** Resource name. */
  name: string;
  /** Resource description. */
  description?: string;
  /** MIME type of the resource. */
  mimeType?: string;
}

/**
 * Resource Read Request parameters.
 * Equivalent to Rust `ReadResourceParams`.
 */
export interface ReadResourceParams {
  /** Resource URI to read. */
  uri: string;
}

/**
 * Resource Read Result.
 * Equivalent to Rust `ReadResourceResult`.
 */
export interface ReadResourceResult {
  /** Resource contents. */
  contents: ResourceContent[];
}

/**
 * Content of a resource (text or binary blob).
 * Equivalent to Rust `ResourceContent`.
 */
export type ResourceContent =
  | { type: "text"; uri: string; mimeType?: string; text: string }
  | { type: "blob"; uri: string; mimeType?: string; blob: string };

/**
 * Resources List Response.
 * Equivalent to Rust `ListResourcesResult`.
 */
export interface ListResourcesResult {
  /** List of available resources. */
  resources: McpResource[];
}

// =============================================================================
// MCP PROMPT TYPES
// =============================================================================

/**
 * MCP Prompt definition.
 * Equivalent to Rust `McpPrompt` (rmcp `Prompt` re-export).
 */
export interface McpPrompt {
  /** Prompt name. */
  name: string;
  /** Prompt description. */
  description?: string;
  /** Prompt arguments. */
  arguments?: PromptArgument[];
}

/**
 * Prompt Argument Definition.
 * Equivalent to Rust `PromptArgument`.
 */
export interface PromptArgument {
  /** Argument name. */
  name: string;
  /** Argument description. */
  description: string;
  /** Whether the argument is required. */
  required: boolean;
}

/**
 * Prompt Get Request parameters.
 * Equivalent to Rust `GetPromptParams`.
 */
export interface GetPromptParams {
  /** Prompt name. */
  name: string;
  /** Optional arguments for the prompt. */
  arguments?: Record<string, unknown>;
}

/**
 * Prompt Get Result.
 * Equivalent to Rust `GetPromptResult`.
 */
export interface GetPromptResult {
  /** Prompt description. */
  description: string;
  /** Prompt messages. */
  messages: PromptMessage[];
}

/**
 * A message within a prompt.
 * Equivalent to Rust `PromptMessage`.
 */
export interface PromptMessage {
  /** Message role (e.g., "user", "assistant"). */
  role: string;
  /** Message content. */
  content: PromptContent;
}

/**
 * Content type within a prompt message.
 * Equivalent to Rust `PromptContent`.
 */
export type PromptContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
  | { type: "resource"; resource: McpResource };

/**
 * Prompts List Response.
 * Equivalent to Rust `ListPromptsResult`.
 */
export interface ListPromptsResult {
  /** List of available prompts. */
  prompts: McpPrompt[];
}
