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

/** Constant-time string equality (no early exit on the first differing byte). */
export function constantTimeEqual(a: string, b: string): boolean {
  const enc = new TextEncoder();
  const x = enc.encode(a);
  const y = enc.encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  }
  return diff === 0;
}

/**
 * Token-based authentication middleware.
 * Skips auth for initialize requests.
 *
 * The transport is one client per process (stdio), so a token presented once
 * authenticates the connection for its lifetime (`ctx.metadata.auth_token`).
 * The `_auth_token` param is removed from the request after it is read so it
 * never reaches the handler or the logging middleware.
 * Equivalent to Rust `AuthMiddleware`.
 */
export class AuthMiddleware implements Middleware {
  private readonly token: string;

  /**
   * Create the middleware for one shared secret.
   * @param token The secret clients must present as the `_auth_token` param.
   */
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
    if (
      typeof storedToken === "string" &&
      constantTimeEqual(storedToken, this.token)
    ) {
      return middlewareContinue();
    }

    // Check params for auth token, then strip it from the request.
    const params = request.params as Record<string, unknown> | undefined;
    if (params && "_auth_token" in params) {
      const paramToken = params["_auth_token"];
      delete params["_auth_token"];
      if (
        typeof paramToken === "string" &&
        constantTimeEqual(paramToken, this.token)
      ) {
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
