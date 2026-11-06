/**
 * @module middleware/tool_filter
 *
 * Tool filtering middleware: static allow/deny lists
 * ({@link ToolFilterMiddleware}) and per-identity policies
 * ({@link IdentityToolFilterMiddleware}).
 * Equivalent to Rust's `ToolFilterMiddleware`.
 */

import type { JsonRpcRequest, JsonRpcResponse } from "@rullama/mcp-client";
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

/**
 * Per-request tool policy for {@link IdentityToolFilterMiddleware}: whether
 * `toolName` is available to the caller `identity` (`ctx.connection.identity`,
 * `undefined` when the connection has none). May be async.
 */
export type IdentityToolPredicate = (
  toolName: string,
  identity: unknown,
  ctx: RequestContext,
) => boolean | Promise<boolean>;

/**
 * Tool filter decided per connection identity: hides the tools a caller may
 * not use from its `tools/list` response and rejects its `tools/call` for them
 * (`-32001`, like {@link ToolFilterMiddleware}). The identity is whatever the
 * listener's `identity` option, a pre-auth hook or a first-line hook put on
 * `ctx.connection.identity` — e.g. one terminal tile may drive only its own
 * tools.
 *
 * @example
 * ```ts ignore
 * const server = new McpServer(handler).withMiddleware(
 *   new IdentityToolFilterMiddleware((tool, identity) =>
 *     (identity as { role?: string } | undefined)?.role === "admin" ||
 *     !tool.startsWith("admin_")
 *   ),
 * );
 * ```
 */
export class IdentityToolFilterMiddleware implements Middleware {
  readonly #allow: IdentityToolPredicate;
  /** Contexts of the `tools/list` requests in flight (each request has its own). */
  readonly #listing = new WeakSet<RequestContext>();

  /**
   * Create a filter from a per-request policy.
   * @param allow Decides, per request, whether the caller may see/call a tool.
   */
  constructor(allow: IdentityToolPredicate) {
    this.#allow = allow;
  }

  /**
   * Shorthand for a fixed allow-list per identity: `toolsFor(identity)`
   * returns the tool names the caller may use, or `"*"` for all of them.
   * @param toolsFor Allowed tool names for an identity (`undefined` = none).
   */
  static allowFor(
    toolsFor: (
      identity: unknown,
      ctx: RequestContext,
    ) => Iterable<string> | "*" | undefined,
  ): IdentityToolFilterMiddleware {
    return new IdentityToolFilterMiddleware((tool, identity, ctx) => {
      const allowed = toolsFor(identity, ctx);
      if (allowed === "*") return true;
      if (allowed === undefined) return false;
      for (const name of allowed) if (name === tool) return true;
      return false;
    });
  }

  /** Whether the caller of `ctx` may use `toolName`. */
  async isAllowed(toolName: string, ctx: RequestContext): Promise<boolean> {
    try {
      return await this.#allow(toolName, ctx.connection?.identity, ctx);
    } catch (e) {
      // Fail closed: a broken policy must not expose tools.
      console.error(`IdentityToolFilterMiddleware predicate failed: ${e}`);
      return false;
    }
  }

  /**
   * Rejects a `tools/call` for a tool this caller may not use, and marks a
   * `tools/list` for filtering in {@link processResponse}.
   */
  async processRequest(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    if (request.method === "tools/list") this.#listing.add(ctx);
    if (request.method !== "tools/call") return middlewareContinue();
    const params = request.params as Record<string, unknown> | undefined;
    const toolName = typeof params?.name === "string" ? params.name : "unknown";
    if (await this.isAllowed(toolName, ctx)) return middlewareContinue();
    return middlewareReject({
      code: -32001,
      message: `Tool '${toolName}' is not allowed for this connection`,
    });
  }

  /** Removes the tools this caller may not use from a `tools/list` result. */
  async processResponse(
    response: JsonRpcResponse,
    ctx: RequestContext,
  ): Promise<void> {
    if (!this.#listing.delete(ctx)) return;
    const result = response.result as { tools?: unknown } | undefined;
    if (!result || !Array.isArray(result.tools)) return;
    const tools = result.tools as { name?: unknown }[];
    const keep = await Promise.all(
      tools.map((t) =>
        typeof t?.name === "string" ? this.isAllowed(t.name, ctx) : false
      ),
    );
    result.tools = tools.filter((_, i) => keep[i]);
  }
}
