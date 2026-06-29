/**
 * @module middleware/rate_limit
 *
 * Token-bucket rate limiting middleware.
 * Equivalent to Rust's `RateLimitMiddleware`.
 */

import type { JsonRpcRequest } from "@rullama/mcp-client";
import type { RequestContext } from "../server.ts";
import {
  type Middleware,
  middlewareContinue,
  middlewareReject,
  type MiddlewareResult,
} from "./mod.ts";

interface RateLimitBucket {
  tokens: number;
  lastRefill: number; // timestamp ms
}

/**
 * Token-bucket rate limiting middleware.
 * Only rate-limits tools/call requests.
 * Equivalent to Rust `RateLimitMiddleware`.
 */
export class RateLimitMiddleware implements Middleware {
  private readonly maxRequestsPerSecond: number;
  private readonly perToolLimits: Map<string, number> = new Map();
  private readonly buckets: Map<string, RateLimitBucket> = new Map();

  constructor(maxRequestsPerSecond: number) {
    this.maxRequestsPerSecond = maxRequestsPerSecond;
  }

  /** Set a per-tool rate limit override. Returns this for chaining. */
  withToolLimit(toolName: string, limit: number): this {
    this.perToolLimits.set(toolName, limit);
    return this;
  }

  private getLimit(key: string): number {
    return this.perToolLimits.get(key) ?? this.maxRequestsPerSecond;
  }

  // deno-lint-ignore require-await
  async processRequest(
    request: JsonRpcRequest,
    _ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    // Only rate-limit tools/call
    if (request.method !== "tools/call") {
      return middlewareContinue();
    }

    const params = request.params as Record<string, unknown> | undefined;
    const toolName = (params?.name as string | undefined) ?? "unknown";

    const limit = this.getLimit(toolName);
    const key = `tool:${toolName}`;

    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = { tokens: limit, lastRefill: Date.now() };
      this.buckets.set(key, bucket);
    }

    // Token bucket refill
    const now = Date.now();
    const elapsed = (now - bucket.lastRefill) / 1000;
    bucket.tokens = Math.min(bucket.tokens + elapsed * limit, limit);
    bucket.lastRefill = now;

    if (bucket.tokens >= 1.0) {
      bucket.tokens -= 1.0;
      return middlewareContinue();
    }

    return middlewareReject({
      code: -32002,
      message: `Rate limited: too many requests for tool '${toolName}'`,
    });
  }
}
