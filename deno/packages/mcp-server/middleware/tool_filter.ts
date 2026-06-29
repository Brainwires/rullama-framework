/**
 * @module middleware/tool_filter
 *
 * Tool filtering middleware (allowlist/blocklist).
 * Equivalent to Rust's `ToolFilterMiddleware`.
 */

import type { JsonRpcRequest } from "@rullama/mcp-client";
import type { RequestContext } from "../server.ts";
import {
  type Middleware,
  middlewareContinue,
  middlewareReject,
  type MiddlewareResult,
} from "./mod.ts";

/** Tool filtering mode. */
export type FilterMode =
  | { type: "allowList"; tools: Set<string> }
  | { type: "denyList"; tools: Set<string> };

/**
 * Middleware that filters tool calls by allow/deny lists.
 * Only applies to tools/call requests.
 * Equivalent to Rust `ToolFilterMiddleware`.
 */
export class ToolFilterMiddleware implements Middleware {
  private readonly mode: FilterMode;

  private constructor(mode: FilterMode) {
    this.mode = mode;
  }

  /** Create a filter that only allows the specified tools. */
  static allowOnly(tools: string[]): ToolFilterMiddleware {
    return new ToolFilterMiddleware({
      type: "allowList",
      tools: new Set(tools),
    });
  }

  /** Create a filter that denies the specified tools. */
  static deny(tools: string[]): ToolFilterMiddleware {
    return new ToolFilterMiddleware({
      type: "denyList",
      tools: new Set(tools),
    });
  }

  /** Check if a tool name is allowed by the current filter. */
  isAllowed(toolName: string): boolean {
    if (this.mode.type === "allowList") {
      return this.mode.tools.has(toolName);
    }
    return !this.mode.tools.has(toolName);
  }

  // deno-lint-ignore require-await
  async processRequest(
    request: JsonRpcRequest,
    _ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    // Only filter tools/call
    if (request.method !== "tools/call") {
      return middlewareContinue();
    }

    const params = request.params as Record<string, unknown> | undefined;
    const toolName = (params?.name as string | undefined) ?? "unknown";

    if (this.isAllowed(toolName)) {
      return middlewareContinue();
    }

    return middlewareReject({
      code: -32001,
      message: `Tool '${toolName}' is not allowed by filter policy`,
    });
  }
}
