/**
 * Model Context Protocol (MCP) client for rullama. Provides {@link McpClient},
 * which spawns MCP servers as subprocesses over a stdio transport, negotiates
 * protocol version `2025-06-18` (and accepts `2025-03-26` / `2024-11-05`), and
 * lists/calls tools, reads resources and fetches prompts. Also exports the
 * JSON-RPC 2.0 and MCP wire types, the {@link StdioTransport} / {@link Transport}
 * layer, and {@link McpConfigManager} for the on-disk server list
 * (`~/.rullama/mcp-config.json`). Equivalent to Rust's `rullama-mcp` crate.
 *
 * @module
 */

// Client
export {
  LATEST_PROTOCOL_VERSION,
  McpClient,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "./client.ts";

// Config
export { McpConfigManager, type McpServerConfig } from "./config.ts";

// Transport
export { StdioTransport, Transport } from "./transport.ts";

// JSON-RPC types (always available)
export type {
  JsonRpcError,
  JsonRpcId,
  JsonRpcMessage,
  JsonRpcNotification,
  JsonRpcRequest,
  JsonRpcResponse,
} from "./types.ts";

export {
  createJsonRpcNotification,
  createJsonRpcRequest,
  isJsonRpcNotification,
  isJsonRpcResponse,
  parseJsonRpcMessage,
  parseNotification,
} from "./types.ts";

// MCP notification types
export type { McpNotification, ProgressParams } from "./types.ts";

// MCP initialization types
export type {
  ClientCapabilities,
  ClientInfo,
  InitializeParams,
  InitializeResult,
  ServerCapabilities,
  ServerInfo,
} from "./types.ts";

// MCP capability types
export type {
  PromptsCapability,
  ResourcesCapability,
  ToolsCapability,
} from "./types.ts";

// MCP tool types
export type {
  CallToolParams,
  CallToolResult,
  Content,
  ListToolsResult,
  McpTool,
  ToolResultContent,
} from "./types.ts";

// MCP resource types
export type {
  ListResourcesResult,
  McpResource,
  ReadResourceParams,
  ReadResourceResult,
  ResourceContent,
} from "./types.ts";

// MCP prompt types
export type {
  GetPromptParams,
  GetPromptResult,
  ListPromptsResult,
  McpPrompt,
  PromptArgument,
  PromptContent,
  PromptMessage,
} from "./types.ts";
