/**
 * @module @rullama/mcp
 *
 * Brainwires MCP - Model Context Protocol client and types.
 * Equivalent to Rust's `rullama-mcp` crate.
 *
 * - **McpClient**: Connect to external MCP servers, list/call tools, resources, prompts
 * - **Transport**: Stdio-based transport layer for MCP communication
 * - **Types**: JSON-RPC 2.0 types and MCP protocol types
 * - **Config**: MCP server configuration management
 */

// Client
export { McpClient } from "./client.ts";

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
