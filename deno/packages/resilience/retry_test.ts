import { assert, assertEquals, assertRejects } from "@std/assert";
import { ChatOptions } from "@brainwires/core";
import {
  applyJitter,
  backoffFor,
  defaultRetryPolicy,
  noRetryPolicy,
  RetryProvider,
  type RetryPolicy,
} from "./retry.ts";
import { ResilienceError } from "./error.ts";
import { EchoProvider } from "./test_util.ts";

Deno.test("default policy is sensible", () => {
  const p = defaultRetryPolicy();
  assertEquals(p.max_attempts, 4);
  assertEquals(p.honor_retry_after, true);
});

Deno.test("backoff doubles then caps", () => {
  const p: RetryPolicy = {
    max_attempts: 10,
    base_ms: 100,
    max_ms: 800,
    jitter: 0,
    honor_retry_after: false,
    overall_deadline_ms: null,
  };
  assertEquals(backoffFor(p, 1), 100);
  assertEquals(backoffFor(p, 2), 200);
  assertEquals(backoffFor(p, 3), 400);
  assertEquals(backoffFor(p, 4), 800);
  assertEquals(backoffFor(p, 5), 800);
});

Deno.test("noRetryPolicy disables retry", () => {
  assertEquals(noRetryPolicy().max_attempts, 1);
});

Deno.test("jitter stays within bounds", () => {
  for (let i = 0; i < 50; i++) {
    const j = applyJitter(1000, 0.2);
    assert(j >= 800 && j <= 1200, `jitter out of bounds: ${j}`);
  }
});

Deno.test("RetryProvider: transient error then success", async () => {
  const inner = EchoProvider.errThenOk("p", 2, "status 503 Service Unavailable");
  const r = new RetryProvider(inner, {
    max_attempts: 5,
    base_ms: 1,
    max_ms: 2,
    jitter: 0,
    honor_retry_after: false,
    overall_deadline_ms: null,
  });
  const resp = await r.chat([], undefined, new ChatOptions());
  assertEquals(resp.message.text(), "ok");
  assertEquals(inner.calls(), 3); // 2 errors + 1 success
});

Deno.test("RetryProvider: non-retryable fails on first attempt", async () => {
  const inner = EchoProvider.alwaysErr("p", "401 Unauthorized");
  const r = new RetryProvider(inner, {
    max_attempts: 5,
    base_ms: 1,
    max_ms: 2,
    jitter: 0,
    honor_retry_after: false,
    overall_deadline_ms: null,
  });
  await assertRejects(() => r.chat([], undefined, new ChatOptions()), Error, "401");
  assertEquals(inner.calls(), 1);
});

Deno.test("RetryProvider: retries exhausted wraps the final error", async () => {
  const inner = EchoProvider.alwaysErr("p", "503 Service Unavailable");
  const r = new RetryProvider(inner, {
    max_attempts: 3,
    base_ms: 1,
    max_ms: 2,
    jitter: 0,
    honor_retry_after: false,
    overall_deadline_ms: null,
  });
  const e = await assertRejects(
    () => r.chat([], undefined, new ChatOptions()),
    ResilienceError,
  );
  assertEquals(e.kind, "retries_exhausted");
  assertEquals(e.detail.attempts, 3);
});
