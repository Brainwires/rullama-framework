# brainwires-telemetry

Unified telemetry for the [Brainwires Agent Framework](https://github.com/Brainwires/brainwires-framework) — analytics events, billing hooks, and cost/usage tracking.

## Overview

`brainwires-telemetry` combines two hook surfaces that always travel together:

- **Analytics** — typed event variants, multi-sink dispatcher, optional SQLite persistence
- **Billing hooks** — `UsageEvent` enum + `BillingHook` trait for pluggable cost tracking

As Nate B Jones put it: *"whoever solves orchestration at infrastructure grade is going to own the most valuable position in the agent stack."* Telemetry is how you instrument that ownership — every token spent, every tool called, every sandbox-second consumed.

## Features

### Analytics

- **`AnalyticsCollector`** — multi-sink dispatcher with typed event variants: `ProviderCall`, `AgentRun`, `ToolCall`, `McpRequest`, `ChannelMessage`, `StorageOp`, `NetworkMessage`, `DreamCycle`, `AutonomySession`, `Custom`
- **`AnalyticsLayer`** — drop-in `tracing-subscriber` layer; intercepts known span names automatically
- **`MemoryAnalyticsSink`** — in-process ring buffer
- **`SqliteAnalyticsSink`** + **`AnalyticsQuery`** (feature `sqlite`) — local SQLite persistence with `cost_by_model()`, `tool_frequency()`, `daily_summary()`

### Billing Hooks

- **`UsageEvent`** — enum covering `Tokens`, `ToolCall`, `SandboxSeconds`, `ApiCall`, `Custom` with `cost_usd()` and `agent_id()` accessors
- **`BillingHook`** — async trait for pluggable billing backends; implement once, wire into `TaskAgentConfig`
- **`BillingError`** — typed error for hook failures

## Usage

```toml
[dependencies]
brainwires-telemetry = { version = "0.11", features = ["sqlite"] }
```

### Analytics

```rust
use brainwires_telemetry::{AnalyticsCollector, MemoryAnalyticsSink, AnalyticsEvent};

let sink = MemoryAnalyticsSink::new(1000);
let collector = AnalyticsCollector::new(vec![Box::new(sink)]);
collector.record(AnalyticsEvent::custom("my_event", serde_json::json!({"key": "value"}))).await;
```

### Billing Hooks

```rust
use brainwires_telemetry::{BillingHook, BillingError, UsageEvent};
use async_trait::async_trait;

struct MyBillingBackend;

#[async_trait]
impl BillingHook for MyBillingBackend {
    async fn on_usage(&self, event: &UsageEvent) -> Result<(), BillingError> {
        println!("agent {} spent ${:.6}", event.agent_id(), event.cost_usd());
        Ok(())
    }
}
```

For a full ledger + wallet implementation with SQLite persistence and Stripe integration, see [`extras/brainwires-billing`](../../extras/brainwires-billing).

## Feature Flags

| Feature  | What it enables |
|----------|-----------------|
| `sqlite` | `SqliteAnalyticsSink` + `AnalyticsQuery` (requires `rusqlite`) |
| `native` | Enables `sqlite` and other native-only features |

## License

MIT OR Apache-2.0
