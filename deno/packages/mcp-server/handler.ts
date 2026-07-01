/**
 * @module handler
 *
 * MCP request handler interface.
 * Equivalent to Rust's `McpHandler` trait.
 */

import type {
  CallToolResult,
  InitializeParams,
  ServerCapabilities,
  ServerInfo,
} from "@rullama/mcp-client";
import type { McpToolDef } from "./registry.ts";
import type { RequestContext } from "./server.ts";

/**
 * Interface for handling MCP protocol requests.
 * Equivalent to Rust `McpHandler` trait.
 */
export interface McpHandler {
  /** Return server identification info. */
  serverInfo(): ServerInfo;

  /** Return server capabilities. */
  capabilities(): ServerCapabilities;

  /** List all available tools. */
  listTools(): McpToolDef[];

  /** Execute a tool call. */
  callTool(
    name: string,
    args: Record<string, unknown>,
    ctx: RequestContext,
  ): Promise<CallToolResult>;

  /** Called when the client sends an initialize request. */
  onInitialize?(params: InitializeParams): Promise<void>;

  /** Called when the server is shutting down. */
  onShutdown?(): Promise<void>;
}
