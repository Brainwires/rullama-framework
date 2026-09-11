# @rullama/call-policy

Provider-layer call policies — retry, budget, circuit breaker, and response
cache. Every policy wraps a `Provider` (from `@rullama/core`) and is itself a
`Provider`, so they compose freely; `ProviderDecorator` is the base class for
writing your own.

## Install

```sh
deno add jsr:@rullama/call-policy
```

## Typical stacking

```ts
import { ChatOptions, Message } from "@rullama/core";
import { AnthropicChatProvider } from "@rullama/provider";
import {
  BudgetGuard,
  BudgetProvider,
  CachedProvider,
  CircuitBreakerProvider,
  defaultCircuitBreakerConfig,
  defaultRetryPolicy,
  ResilienceError,
  RetryProvider,
} from "@rullama/call-policy";

const base = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-6",
);

// Innermost first: cache → budget → retry → circuit breaker.
const cached = CachedProvider.withMemoryCache(base).provider;
const guard = new BudgetGuard({
  max_tokens: 250_000,
  max_usd_cents: null,
  max_rounds: 30,
});
const budgeted = new BudgetProvider(cached, guard);
const retried = new RetryProvider(budgeted, defaultRetryPolicy());
const provider = new CircuitBreakerProvider(
  retried,
  defaultCircuitBreakerConfig(),
).withFallback(base);

try {
  const resp = await provider.chat(
    [Message.user("Summarise the release notes.")],
    undefined,
    new ChatOptions({ max_tokens: 512 }),
  );
  console.log(resp.message.text(), guard.tokensConsumed(), "tokens so far");
} catch (err) {
  if (err instanceof ResilienceError) {
    // "budget_exceeded" | "circuit_open" | "retries_exhausted" | "deadline_exceeded"
    console.error(err.kind, err.detail);
  } else throw err;
}
```

Outermost first: `CircuitBreaker → Retry → Budget → Cache → base`.

## What each decorator does

- **RetryProvider** — exponential backoff with proportional jitter on transient
  failures. Errors are classified by string-matching their message
  (`classifyError`): `rate_limited`, `network` and `server_5xx` are retried;
  `auth`, `client_4xx` and `unknown` are rethrown immediately. A
  `retry-after: N` hint in the message is honored when `honor_retry_after` is
  set, and `overall_deadline_ms` caps the whole sequence. Streaming bypasses
  retry — partial streams can't be safely replayed.
- **BudgetProvider** — caps on tokens, USD cents, and rounds held in a shared
  `BudgetGuard` (share one guard between providers for a joint budget). Every
  call is pre-flight checked, counted as a round, and its reported `usage`
  accumulated; a call is rejected up front when its estimated input tokens alone
  would push past `max_tokens`. USD spend is only tracked if you call
  `guard.recordCostCents(...)` yourself — the decorator does not price
  responses. Streaming is budgeted too (via `usage` chunks).
- **CircuitBreakerProvider** — closed / open / half-open state machine keyed by
  `options.model` (or a custom `modelKey`) under the provider name. After
  `failure_threshold` consecutive failures the circuit opens for `cooldown_ms`;
  the next call after that is a probe. While open, calls go to the fallback set
  with `withFallback(...)`, or throw `circuit_open`. Streaming bypasses the
  breaker.
- **CachedProvider** — SHA-256 content-addressed cache over messages, tool names
  (sorted) and options; pass a `scope` to partition entries per tenant or user.
  Any `CacheBackend` implementation works — `MemoryCache` (bounded LRU, 1000
  entries by default) ships here; bring your own for Deno KV / Postgres / Redis.
  Cache writes that fail are ignored. Streaming bypasses the cache.

Every policy failure is a `ResilienceError` whose `kind` discriminates the cause
and whose `detail` carries the numbers (`consumed`/`limit`, `attempts`,
`failures`, …); the last underlying error is attached as `cause` for the retry
kinds.

## Equivalent Rust crate

`rullama-resilience` — same decorator shapes, same semantics. The Rust crate's
optional SQLite cache backend is intentionally not ported; implement
`CacheBackend` directly against Deno KV (or any other store) for persistence.
