/**
 * Tests for the middleware chain.
 * Equivalent to Rust middleware::tests.
 */

import { assertEquals } from "@std/assert";
import type { JsonRpcRequest } from "@rullama/mcp-client";
import { RequestContext } from "../server.ts";
import {
  type Middleware,
  MiddlewareChain,
  middlewareContinue,
  middlewareReject,
  type MiddlewareResult,
} from "./mod.ts";

class PassMiddleware implements Middleware {
  // deno-lint-ignore require-await
  async processRequest(
    _request: JsonRpcRequest,
    _ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    return middlewareContinue();
  }
}

class RejectMiddleware implements Middleware {
  // deno-lint-ignore require-await
  async processRequest(
    _request: JsonRpcRequest,
    _ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    return middlewareReject({
      code: -32003,
      message: "Rejected",
    });
  }
}

const makeRequest = (method = "test"): JsonRpcRequest => ({
  jsonrpc: "2.0",
  id: 1,
  method,
});

Deno.test("middleware chain - all pass", async () => {
  const chain = new MiddlewareChain();
  chain.add(new PassMiddleware());
  chain.add(new PassMiddleware());

  const ctx = new RequestContext(1);
  const result = await chain.processRequest(makeRequest(), ctx);
  assertEquals(result, null);
});

Deno.test("middleware chain - reject stops chain", async () => {
  const chain = new MiddlewareChain();
  chain.add(new PassMiddleware());
  chain.add(new RejectMiddleware());
  chain.add(new PassMiddleware());

  const ctx = new RequestContext(1);
  const result = await chain.processRequest(makeRequest(), ctx);
  assertEquals(result?.code, -32003);
});

Deno.test("middleware chain - empty chain passes", async () => {
  const chain = new MiddlewareChain();
  const ctx = new RequestContext(1);
  const result = await chain.processRequest(makeRequest(), ctx);
  assertEquals(result, null);
});
