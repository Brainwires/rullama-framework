# brainwires-call-policy

Provider-layer resilience middleware for the Brainwires Agent Framework.
Composable decorators that wrap any `brainwires_core::Provider` with retry
logic, cost/token/round budgets, and a circuit breaker — without changing
the wrapped provider's call signature.

```
 CircuitBreaker → Retry → Budget → base Provider
```

Each decorator takes `Arc<dyn Provider>` and returns something that
implements `Provider`, so they compose in any order. The recommended
outer→inner ordering above fails fast when the circuit is open (skipping
both retries and budget charges), avoids charging budget for calls that
will be retried internally, and lets the base provider do a single real
call per unit of retry work.

## Features

| Flag      | Default | Enables                                    |
|-----------|---------|--------------------------------------------|
| `native`  | on      | Core decorators (retry, budget, breaker).  |
| `cache`   | off     | `SqliteCache` backend for `CachedProvider`.|

## Quick start

```rust
use std::sync::Arc;
use std::time::Duration;
use brainwires_resilience::{
    BudgetConfig, BudgetGuard, BudgetProvider,
    CircuitBreakerConfig, CircuitBreakerProvider,
    RetryPolicy, RetryProvider,
};

let base: Arc<dyn brainwires_core::Provider> = /* your provider */;

let budgeted = Arc::new(BudgetProvider::new(
    base,
    BudgetGuard::new(BudgetConfig {
        max_usd_cents: Some(1_000),
        max_tokens: Some(500_000),
        max_rounds: Some(50),
    }),
));

let retried = Arc::new(RetryProvider::new(
    budgeted,
    RetryPolicy {
        max_attempts: 4,
        base: Duration::from_millis(500),
        max: Duration::from_secs(30),
        jitter: 0.2,
        honor_retry_after: true,
        overall_deadline: Some(Duration::from_secs(60)),
    },
));

let guarded: Arc<dyn brainwires_core::Provider> = Arc::new(
    CircuitBreakerProvider::new(
        retried,
        CircuitBreakerConfig {
            failure_threshold: 5,
            cooldown: Duration::from_secs(30),
        },
    ),
);
```

## API

### `RetryProvider`

Retries non-streaming `chat` calls classified as transient (HTTP 429/5xx,
network errors, `Retry-After` hints). Never retries streaming calls —
partial streams cannot be safely replayed.

```rust
pub struct RetryPolicy {
    pub max_attempts: u32,           // total attempts, 1 disables retry
    pub base: Duration,              // initial backoff
    pub max: Duration,               // single-attempt sleep cap
    pub jitter: f64,                 // ±proportional jitter, 0.0..=1.0
    pub honor_retry_after: bool,     // parse upstream Retry-After hints
    pub overall_deadline: Option<Duration>, // hard ceiling for the loop
}
```

Failure modes:
- `ResilienceError::RetriesExhausted { attempts, source }` — `max_attempts`
  reached with transient errors.
- `ResilienceError::DeadlineExceeded { attempts, elapsed_ms, source }` —
  `overall_deadline` tripped before a success. Each scheduled sleep is
  also capped to the remaining deadline so no single wait overshoots.

### `BudgetProvider`

Atomic counters (no locks) for USD cents, tokens, and rounds. Pre-flight
checks before each call; post-flight accumulation from `Usage`.

```rust
pub struct BudgetConfig {
    pub max_usd_cents: Option<u64>,
    pub max_tokens:    Option<u64>,
    pub max_rounds:    Option<u64>,
}
```

Returns `ResilienceError::BudgetExceeded { kind, consumed, limit }` when a
cap is hit pre-flight. A `BudgetGuard` is cheaply `Clone`able — share one
across multiple providers to enforce a global cap, or per-provider for
isolated caps.

### `CircuitBreakerProvider`

Closed → Open → HalfOpen state machine keyed by `(provider, model)`:

- **Closed** — every call goes through. Consecutive failures increment a
  counter; hitting `failure_threshold` trips the breaker to **Open**.
- **Open** — calls fail fast with `ResilienceError::CircuitOpen` until
  `cooldown` elapses, then the next call enters **HalfOpen**.
- **HalfOpen** — exactly one trial call is permitted. Success → Closed;
  failure → Open again (with the cooldown restarting).

### `CachedProvider`

In-memory (`MemoryCache`) or SQLite-backed (`SqliteCache`, feature `cache`)
response cache keyed on `cache_key_for(messages, tools, options)`. Bypass
on high-temperature or streaming calls via `options.skip_cache`.

## Errors

```rust
pub enum ResilienceError {
    BudgetExceeded   { kind, consumed, limit },
    CircuitOpen      { provider, model, failures },
    RetriesExhausted { attempts, source },
    DeadlineExceeded { attempts, elapsed_ms, source },
}
```

All variants ride through the `Provider` trait as `anyhow::Error`; recover
the typed form with `err.downcast_ref::<ResilienceError>()`.

## Error classification

`classify_error()` inspects the error's string form for transient markers
(`429`, `500`, `503`, `timeout`, `connection`, etc.). This is intentional
— providers surface errors as strings and we don't want to couple to any
single provider's typed error hierarchy. The taxonomy:

- `Transient` — retryable (default).
- `Permanent` — 4xx non-retryable (auth, validation).
- `Fatal` — non-network (serialisation, panic).

Only `Transient` is retried.

## Composition notes

- **Retry before Budget** doubles the token spend on every retry. Place
  Budget *inside* Retry so only the final attempt's `Usage` accumulates.
- **Circuit outside Retry** avoids burning retry budget during a known
  outage. The breaker's `failure_threshold` should be ≥ Retry's
  `max_attempts` or every exhausted-retry chain trips the circuit on the
  first cascade.
- `stream_chat` passes through every decorator unchanged.

## Status

The decorators are stable; integration into the default `ChatAgent`
construction path is opt-in via `ChatAgentBuilder::with_resilience` (in
`brainwires-agent`). This crate is `#[deny(missing_docs)]`.
