/**
 * @module middleware
 *
 * Middleware pipeline for MCP request/response processing.
 * Equivalent to Rust's `Middleware`, `MiddlewareChain`, `MiddlewareResult`.
 */

import type {
  JsonRpcError,
  JsonRpcRequest,
  JsonRpcResponse,
} from "@rullama/mcp-client";
import type { RequestContext } from "../server.ts";

/**
 * Result of middleware processing.
 * Equivalent to Rust `MiddlewareResult`.
 */
export type MiddlewareResult =
  | { type: "continue" }
  | { type: "reject"; error: JsonRpcError };

/** Create a Continue result. */
export function middlewareContinue(): MiddlewareResult {
  return { type: "continue" };
}

/** Create a Reject result. */
export function middlewareReject(error: JsonRpcError): MiddlewareResult {
  return { type: "reject", error };
}

/**
 * Interface for request/response middleware.
 * Equivalent to Rust `Middleware` trait.
 */
export interface Middleware {
  /** Process an incoming request. Return Continue or Reject. */
  processRequest(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<MiddlewareResult>;

  /** Optionally process the outgoing response. */
  processResponse?(
    response: JsonRpcResponse,
    ctx: RequestContext,
  ): Promise<void>;
}

/**
 * Ordered chain of middleware layers.
 * Equivalent to Rust `MiddlewareChain`.
 */
export class MiddlewareChain {
  private layers: Middleware[] = [];

  /** Add a middleware layer to the chain. */
  add(middleware: Middleware): void {
    this.layers.push(middleware);
  }

  /**
   * Run all middleware on the request, stopping on first reject.
   * Returns the rejection error or null if all passed.
   */
  async processRequest(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcError | null> {
    for (const layer of this.layers) {
      const result = await layer.processRequest(request, ctx);
      if (result.type === "reject") {
        return result.error;
      }
    }
    return null;
  }

  /** Run all middleware on the response. */
  async processResponse(
    response: JsonRpcResponse,
    ctx: RequestContext,
  ): Promise<void> {
    for (const layer of this.layers) {
      if (layer.processResponse) {
        await layer.processResponse(response, ctx);
      }
    }
  }
}

export { AuthMiddleware } from "./auth.ts";
export { LoggingMiddleware } from "./logging.ts";
export { RateLimitMiddleware } from "./rate_limit.ts";
export { ToolFilterMiddleware } from "./tool_filter.ts";
