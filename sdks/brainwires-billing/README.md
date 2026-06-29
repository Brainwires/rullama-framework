# brainwires-billing-impl

Full billing implementation for the [Brainwires Agent Framework](https://github.com/Brainwires/brainwires-framework) — ledger storage, per-customer wallet, and Stripe integration.

## Overview

This crate implements the hook surface defined in `crates/brainwires-billing`. The framework crate stays thin (just `UsageEvent` + `BillingHook` trait); all storage and payment logic lives here.

## Components

- **`BillingLedger`** — async trait for pluggable event storage
- **`InMemoryLedger`** — in-process, zero-dependency ledger for tests
- **`SqliteLedger`** *(feature `sqlite`)* — WAL-mode SQLite at `~/.brainwires/billing/billing.db`
- **`AgentWallet`** — implements `BillingHook`; accumulates per-customer spend, enforces a USD budget ceiling, persists every event to a ledger
- **`StripeClient`** *(feature `stripe`)* — reports metered usage, creates payment links, queries customer balance

## Usage

```toml
[dependencies]
brainwires-billing-impl = { path = "extras/brainwires-billing" }
brainwires-agent = { path = "crates/brainwires-agent", features = ["billing"] }
```

```rust
use brainwires_billing_impl::{AgentWallet, SqliteLedger};
use brainwires_agent::task_agent::{BillingHookRef, TaskAgentConfig};
use std::sync::Arc;

// $5.00 budget per customer session
let ledger = Arc::new(SqliteLedger::new_default()?);
let wallet = AgentWallet::new("customer-42".into(), Some(5.00), ledger);

let config = TaskAgentConfig {
    billing_hook: Some(BillingHookRef::new(wallet)),
    ..Default::default()
};
```

Every provider call and tool call the agent makes fires `on_usage()` on the wallet, which persists the event and checks the budget. Errors are logged but never abort the run (fail-open) — check `wallet.budget_exhausted()` between iterations if you want hard enforcement.

## Advisory vs enforced hooks

`BillingHook` exposes two methods with different failure semantics:

| Method | When called | On error | Intended use |
|---|---|---|---|
| `on_usage(event)` | **After** a call has happened | Logged, call already completed | Ledger persistence, analytics, metered billing |
| `authorize(pending)` | **Before** a call is dispatched | Tool call is rejected (fail-closed) | Hard budget enforcement |

`authorize()` has a default implementation that returns `Ok(())`, so existing `BillingHook` integrators who only care about observation do not need to change any code — their hooks remain purely advisory.

`AgentWallet` overrides `authorize()` to return `BillingError::BudgetExhausted` as soon as `wallet.budget_exhausted()` is true, which causes the agent's tool-call dispatcher to reject the pending call before it runs. The advisory `on_usage()` path is unchanged: it still records every event to the ledger and logs a `Hook(...)` error when the ceiling is crossed, for integrators who want to observe but not enforce.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `native` | ✅ | Enables `sqlite` + directory detection via `dirs` |
| `sqlite` | ✅ | SQLite-backed `SqliteLedger` |
| `stripe` | ❌ | Stripe REST client (`reqwest` dependency) |

## License

MIT OR Apache-2.0
