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
import type { ConnectionInfo } from "./transport/traits.ts";

/**
 * Interface for handling MCP protocol requests.
 * Equivalent to Rust `McpHandler` trait.
 */
export interface McpHandler {
  /** Return server identification info. */
  serverInfo(): ServerInfo;

  /** Return server capabilities. */
  capabilities(): ServerCapabilities;

  /**
   * List all available tools. `ctx` identifies the connection asking
   * (`ctx.connection`), so one handler can serve per-connection tool lists.
   */
  listTools(ctx?: RequestContext): McpToolDef[];

  /**
   * Execute a tool call. `ctx.signal` aborts when the client cancels the
   * request or disconnects; `ctx.connection` says which connection it came on.
   */
  callTool(
    name: string,
    args: Record<string, unknown>,
    ctx: RequestContext,
  ): Promise<CallToolResult>;

  /** Called when the client sends an initialize request (`ctx` is that request's context). */
  onInitialize?(params: InitializeParams, ctx?: RequestContext): Promise<void>;

  /**
   * Called when a connection served by this handler ends (once per
   * connection; with stdio that is server shutdown).
   */
  onShutdown?(connection?: ConnectionInfo): Promise<void>;
}
