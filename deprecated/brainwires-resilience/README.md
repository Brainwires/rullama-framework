# brainwires-resilience (DEPRECATED)

This crate has been **renamed** to
[`brainwires-call-policy`](https://crates.io/crates/brainwires-call-policy).

The old name `resilience` was abstract — borrowed from the Resilience4j /
Polly tradition. The crate's actual content is policies applied to
outbound provider calls (retry-with-backoff, circuit breaker, budget
caps, response cache, error classification). The new name says that.

There is no re-export shim — depending on this crate gets you nothing.

## Migration

```toml
# Before
brainwires-resilience = "0.10"

# After
brainwires-call-policy = "0.11"
```

```rust
// Before
use brainwires_resilience::{Retry, CircuitBreaker, Budget, Cache, classify};

// After
use brainwires_call_policy::{Retry, CircuitBreaker, Budget, Cache, classify};
```

The public API is otherwise unchanged.
