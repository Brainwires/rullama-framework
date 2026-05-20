#![deprecated(
    since = "0.10.1",
    note = "renamed to `brainwires-call-policy`. The crate's content (retry / circuit breaker / budget / cache / error classification) is policies applied to outbound provider calls; the new name says that, the old `resilience` was abstract. There is no re-export shim — switch your dep and your imports."
)]
//! `brainwires-resilience` is **deprecated** as of 0.10.1.
//!
//! Renamed to [`brainwires-call-policy`](https://crates.io/crates/brainwires-call-policy)
//! — the crate's content is policies you apply to outbound provider calls
//! (retry-with-backoff, circuit breaker, budget caps, response cache,
//! error classification), and the new name says that.
//!
//! There is no re-export shim.
