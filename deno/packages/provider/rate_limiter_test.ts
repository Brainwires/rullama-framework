import { assertEquals, assertThrows } from "@std/assert";
import { assert } from "@std/assert/assert";
import { RateLimitedClient, RateLimiter } from "./rate_limiter.ts";

Deno.test("RateLimiter: creation sets correct values", () => {
  const limiter = new RateLimiter(60);
  assertEquals(limiter.maxRequestsPerMinute(), 60);
  assertEquals(limiter.availableTokens(), 60);
});

Deno.test("RateLimiter: zero rpm is rejected (it could never be satisfied)", () => {
  assertThrows(() => new RateLimiter(0), Error, "must be positive");
});

Deno.test("RateLimiter: acquire consumes token", async () => {
  const limiter = new RateLimiter(10);
  assertEquals(limiter.availableTokens(), 10);

  await limiter.acquire();
  assertEquals(limiter.availableTokens(), 9);

  await limiter.acquire();
  assertEquals(limiter.availableTokens(), 8);
});

Deno.test("RateLimiter: multiple acquires drain tokens", async () => {
  const limiter = new RateLimiter(5);

  for (let i = 0; i < 5; i++) {
    await limiter.acquire();
  }

  assertEquals(limiter.availableTokens(), 0);
});

Deno.test("RateLimiter: tryAcquire returns true when tokens available", () => {
  const limiter = new RateLimiter(3);
  assert(limiter.tryAcquire());
  assertEquals(limiter.availableTokens(), 2);
});

Deno.test("RateLimiter: tryAcquire returns false when no tokens", async () => {
  const limiter = new RateLimiter(1);
  await limiter.acquire();
  assertEquals(limiter.tryAcquire(), false);
});

Deno.test("RateLimitedClient: wraps function with rate limiting", async () => {
  let callCount = 0;
  // deno-lint-ignore require-await
  const fn = async (x: number): Promise<number> => {
    callCount++;
    return x * 2;
  };

  const client = new RateLimitedClient(fn, { requestsPerMinute: 100 });
  const result = await client.execute(21);

  assertEquals(result, 42);
  assertEquals(callCount, 1);
  assertEquals(client.limiter.availableTokens(), 99);
});

Deno.test("RateLimitedClient: multiple calls consume tokens", async () => {
  // deno-lint-ignore require-await
  const fn = async (s: string): Promise<string> => `hello ${s}`;
  const client = new RateLimitedClient(fn, { requestsPerMinute: 5 });

  const results = [];
  for (let i = 0; i < 3; i++) {
    results.push(await client.execute("world"));
  }

  assertEquals(results.length, 3);
  assertEquals(results[0], "hello world");
  assertEquals(client.limiter.availableTokens(), 2);
});
