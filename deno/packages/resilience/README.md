# @brainwires/resilience

Provider-layer resilience middleware — retry, budget, circuit-breaker, and
response cache. Every decorator wraps a `Provider` (from `@brainwires/core`)
and returns a `Provider`, so they compose freely.

## Typical stacking

```ts
import { AnthropicChatProvider } from "@brainwires/providers";
import {
  BudgetGuard,
  BudgetProvider,
  CachedProvider,
  CircuitBreakerProvider,
  defaultRetryPolicy,
  RetryProvider,
} from "@brainwires/resilience";

const base = new AnthropicChatProvider(key, "claude-sonnet-4-6", "anthropic");

const cached = CachedProvider.withMemoryCache(base).provider;
const budgeted = new BudgetProvider(cached, new BudgetGuard({
  max_tokens: 250_000,
  max_usd_cents: null,
  max_rounds: 30,
}));
const retried = new RetryProvider(budgeted, defaultRetryPolicy());
const provider = new CircuitBreakerProvider(retried);
```

Outermost first: `CircuitBreaker → Retry → Budget → Cache → base`.

## What each decorator does

- **RetryProvider** — exponential backoff with jitter on transient failures
  (429 / 5xx / network). Honors `retry-after` hints in error messages.
  Streaming bypasses retry — partial streams can't be safely replayed.
- **BudgetProvider** — atomic caps on tokens, USD cents, and rounds. Pre-flight
  rejection when a payload alone would push past the token cap.
- **CircuitBreakerProvider** — half-open state machine with an optional
  fallback provider. Streaming bypasses the breaker.
- **CachedProvider** — SHA-256 content-addressed cache. Tools are name-sorted
  before hashing. Any `CacheBackend` implementation works (in-memory ships
  here; bring your own for Deno KV / Postgres / Redis). Streaming bypasses
  the cache.

## Equivalent Rust crate

`brainwires-resilience` — same decorator shapes, same semantics. The Rust
crate's optional SQLite cache backend is intentionally not ported; implement
`CacheBackend` directly against Deno KV (or any other store) for persistence.
