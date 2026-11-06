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
  /** Optional display name (MCP 2025-06-18+). */
  title?: string;
  /** Optional JSON Schema of `structuredContent` results (MCP 2025-06-18+). */
  outputSchema?: Record<string, unknown>;
  /** Optional behaviour hints (`readOnlyHint`, `destructiveHint`, …). */
  annotations?: Record<string, unknown>;
  /** Optional icons (`{ src, mimeType?, sizes?, theme? }`, MCP 2025-11-25). */
  icons?: Record<string, unknown>[];
  /** Optional protocol metadata passed through unchanged. */
  _meta?: Record<string, unknown>;
}

/**
 * MCP `ToolAnnotations`: hints about a tool's behaviour for clients (e.g. to
 * decide what needs confirmation). Hints are advisory and untrusted: a
 * client must not rely on them from a server it does not trust.
 */
export interface McpToolAnnotations {
  /** Display name for the tool (`Tool.title` takes precedence). */
  title?: string;
  /** The tool does not modify its environment (default `false`). */
  readOnlyHint?: boolean;
  /** When not read-only: updates may be destructive rather than purely
   * additive (default `true`). */
  destructiveHint?: boolean;
  /** When not read-only: repeating a call with the same arguments has no
   * further effect (default `false`). */
  idempotentHint?: boolean;
  /** The tool talks to an open world of external entities, e.g. the web
   * (default `true`). */
  openWorldHint?: boolean;
  /** Hints from later protocol revisions pass through unchanged. */
  [key: string]: unknown;
}

/**
 * Optional `Tool` fields for {@link McpToolRegistry.register}; each one set is
 * emitted in `tools/list`.
 */
export interface McpToolOptions {
  /** Display name (MCP 2025-06-18+). */
  title?: string;
  /** Behaviour hints (`readOnlyHint`, `destructiveHint`, …). */
  annotations?: McpToolAnnotations;
  /** JSON Schema of `structuredContent` results (MCP 2025-06-18+). */
  outputSchema?: Record<string, unknown>;
  /** Icons (`{ src, mimeType?, sizes?, theme? }`, MCP 2025-11-25). */
  icons?: Record<string, unknown>[];
  /** Protocol metadata passed through unchanged. */
  _meta?: Record<string, unknown>;
}

/**
 * Function type for tool execution handlers.
 * Equivalent to Rust `ToolHandler` trait.
 */
export type ToolHandler = (
  args: Record<string, unknown>,
  ctx: RequestContext,
) => Promise<CallToolResult>;

/** The `McpToolOptions` keys copied onto the definition. */
const OPTION_KEYS: ReadonlySet<string> = new Set([
  "title",
  "annotations",
  "outputSchema",
  "icons",
  "_meta",
]);

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

  /**
   * Register a tool with its handler.
   *
   * @param options Optional `Tool` fields (`annotations`, `title`,
   *   `outputSchema`, `icons`, `_meta`); the ones set appear in `listTools()`
   *   and so in `tools/list`.
   *
   * @example
   * ```ts
   * registry.register("read_file", "Read a file", schema, handler, {
   *   annotations: { readOnlyHint: true, openWorldHint: false },
   * });
   * ```
   */
  register(
    name: string,
    description: string,
    inputSchema: Record<string, unknown>,
    handler: ToolHandler,
    options: McpToolOptions = {},
  ): void {
    const extra: McpToolOptions = Object.fromEntries(
      Object.entries(options).filter(([key, value]) =>
        value !== undefined && OPTION_KEYS.has(key)
      ),
    );
    this.tools.push({
      def: { ...extra, name, description, inputSchema },
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
