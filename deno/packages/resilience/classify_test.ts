import { assert, assertEquals } from "@std/assert";
import { classifyError, isRetryable, parseRetryAfter } from "./classify.ts";

Deno.test("classify 429", () => {
  const err = new Error("status 429 Too Many Requests");
  assertEquals(classifyError(err), "rate_limited");
  assert(isRetryable(classifyError(err)));
});

Deno.test("classify 5xx", () => {
  const err = new Error("status 503 Service Unavailable");
  assertEquals(classifyError(err), "server_5xx");
});

Deno.test("classify network", () => {
  const err = new Error("connection reset by peer");
  assertEquals(classifyError(err), "network");
});

Deno.test("classify auth not retryable", () => {
  const err = new Error("401 Unauthorized: invalid API key");
  assertEquals(classifyError(err), "auth");
  assert(!isRetryable(classifyError(err)));
});

Deno.test("parse retry-after seconds → ms", () => {
  const err = new Error("429 Too Many Requests; retry-after: 42");
  assertEquals(parseRetryAfter(err), 42_000);
});

Deno.test("parse retry-after absent", () => {
  assertEquals(parseRetryAfter(new Error("generic failure")), null);
});

Deno.test("parse retry-after rejects absurd", () => {
  assertEquals(parseRetryAfter(new Error("retry-after: 99999")), null);
});
