/**
 * @module registry
 *
 * MCP tool registry with handler dispatch.
 * Equivalent to Rust's `McpToolRegistry`, `ToolHandler`, `McpToolDef`.
 */

import type { CallToolResult } from "@rullama/mcp-client";
import type { RequestContext } from "./server.ts";
import { AgentNetworkError } from "./error.ts";

/**
 * Definition of an MCP tool.
 * Equivalent to Rust `McpToolDef`.
 */
export interface McpToolDef {
  /** Tool name. */
  name: string;
  /** Human-readable description. */
  description: string;
  /** JSON Schema for tool input. */
  inputSchema: Record<string, unknown>;
}

/**
 * Function type for tool execution handlers.
 * Equivalent to Rust `ToolHandler` trait.
 */
export type ToolHandler = (
  args: Record<string, unknown>,
  ctx: RequestContext,
) => Promise<CallToolResult>;

interface RegisteredTool {
  def: McpToolDef;
  handler: ToolHandler;
}

/**
 * Registry of MCP tools with their handlers.
 * Equivalent to Rust `McpToolRegistry`.
 */
export class McpToolRegistry {
  private tools: RegisteredTool[] = [];

  /** Register a tool with its handler. */
  register(
    name: string,
    description: string,
    inputSchema: Record<string, unknown>,
    handler: ToolHandler,
  ): void {
    this.tools.push({
      def: { name, description, inputSchema },
      handler,
    });
  }

  /** List all registered tool definitions. */
  listTools(): McpToolDef[] {
    return this.tools.map((t) => ({ ...t.def }));
  }

  /** Dispatch a tool call to its registered handler. */
  // deno-lint-ignore require-await
  async dispatch(
    name: string,
    args: Record<string, unknown>,
    ctx: RequestContext,
  ): Promise<CallToolResult> {
    const tool = this.tools.find((t) => t.def.name === name);
    if (!tool) {
      throw AgentNetworkError.toolNotFound(name);
    }
    return tool.handler(args, ctx);
  }

  /** Check if a tool is registered. */
  hasTool(name: string): boolean {
    return this.tools.some((t) => t.def.name === name);
  }
}
