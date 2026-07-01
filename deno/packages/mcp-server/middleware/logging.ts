/**
 * @module middleware/logging
 *
 * Request/response logging middleware.
 * Equivalent to Rust's `LoggingMiddleware`.
 */

import type { JsonRpcRequest, JsonRpcResponse } from "@rullama/mcp-client";
import type { RequestContext } from "../server.ts";
import {
  type Middleware,
  middlewareContinue,
  type MiddlewareResult,
} from "./mod.ts";

/**
 * Middleware that logs all requests and responses.
 * Equivalent to Rust `LoggingMiddleware`.
 */
export class LoggingMiddleware implements Middleware {
  // deno-lint-ignore require-await
  async processRequest(
    request: JsonRpcRequest,
    _ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    console.debug(
      `[MCP] request received: method=${request.method} id=${
        JSON.stringify(request.id)
      }`,
    );
    return middlewareContinue();
  }

  // deno-lint-ignore require-await
  async processResponse(
    response: JsonRpcResponse,
    _ctx: RequestContext,
  ): Promise<void> {
    if (response.error) {
      console.warn(
        `[MCP] response with error: id=${JSON.stringify(response.id)} error=${
          JSON.stringify(response.error)
        }`,
      );
    } else {
      console.debug(
        `[MCP] response sent: id=${JSON.stringify(response.id)}`,
      );
    }
  }
}
