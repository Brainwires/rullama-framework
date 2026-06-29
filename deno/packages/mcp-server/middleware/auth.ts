/**
 * @module middleware/auth
 *
 * Token-based authentication middleware.
 * Equivalent to Rust's `AuthMiddleware`.
 */

import type { JsonRpcRequest } from "@rullama/mcp-client";
import type { RequestContext } from "../server.ts";
import {
  type Middleware,
  middlewareContinue,
  middlewareReject,
  type MiddlewareResult,
} from "./mod.ts";

/**
 * Token-based authentication middleware.
 * Skips auth for initialize requests.
 * Equivalent to Rust `AuthMiddleware`.
 */
export class AuthMiddleware implements Middleware {
  private readonly token: string;

  constructor(token: string) {
    this.token = token;
  }

  // deno-lint-ignore require-await
  async processRequest(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    // Skip auth for initialize - clients haven't authenticated yet
    if (request.method === "initialize") {
      return middlewareContinue();
    }

    // Check for token in metadata (set during initialize)
    const storedToken = ctx.metadata.get("auth_token");
    if (typeof storedToken === "string" && storedToken === this.token) {
      return middlewareContinue();
    }

    // Check params for auth token
    const params = request.params as Record<string, unknown> | undefined;
    if (params) {
      const paramToken = params["_auth_token"];
      if (typeof paramToken === "string" && paramToken === this.token) {
        ctx.metadata.set("auth_token", paramToken);
        return middlewareContinue();
      }
    }

    return middlewareReject({
      code: -32003,
      message: "Unauthorized: invalid or missing auth token",
    });
  }
}
