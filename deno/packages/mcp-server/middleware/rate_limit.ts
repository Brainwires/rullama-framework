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

/** Buckets kept before the least-recently-used one is dropped. */
export const MAX_RATE_LIMIT_BUCKETS = 1024;
/** Longest tool name that gets its own bucket; longer names share one. */
const MAX_KEY_TOOL_NAME = 128;

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

  /** Get-or-create the bucket for `key`, refreshing its LRU position. */
  private bucketFor(key: string, limit: number): RateLimitBucket {
    const existing = this.buckets.get(key);
    if (existing) this.buckets.delete(key); // refresh recency
    const bucket = existing ?? { tokens: limit, lastRefill: Date.now() };
    this.buckets.set(key, bucket);
    while (this.buckets.size > MAX_RATE_LIMIT_BUCKETS) {
      this.buckets.delete(this.buckets.keys().next().value as string);
    }
    return bucket;
  }

  /** Refill by elapsed time, then take one token; false when none is left. */
  private static take(bucket: RateLimitBucket, limit: number): boolean {
    const now = Date.now();
    const elapsed = (now - bucket.lastRefill) / 1000;
    bucket.tokens = Math.min(bucket.tokens + elapsed * limit, limit);
    bucket.lastRefill = now;
    if (bucket.tokens < 1.0) return false;
    bucket.tokens -= 1.0;
    return true;
  }

  processRequest(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    // Only rate-limit tools/call
    if (request.method !== "tools/call") {
      return Promise.resolve(middlewareContinue());
    }
    const params = request.params as Record<string, unknown> | undefined;
    const rawName = params?.name;
    const toolName =
      typeof rawName === "string" && rawName.length <= MAX_KEY_TOOL_NAME
        ? rawName
        : "unknown";
    const limit = this.getLimit(toolName);
    // Keyed per client so one client cannot exhaust another's budget. The map
    // is bounded (LRU) because the tool name is attacker-controlled input.
    const key = `${ctx.clientInfo?.name ?? "anonymous"}:${toolName}`;
    if (RateLimitMiddleware.take(this.bucketFor(key, limit), limit)) {
      return Promise.resolve(middlewareContinue());
    }
    return Promise.resolve(middlewareReject({
      code: -32002,
      message: `Rate limited: too many requests for tool '${toolName}'`,
    }));
  }
}
