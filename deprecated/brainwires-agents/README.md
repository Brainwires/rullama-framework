# brainwires-agents (DEPRECATED)

This crate has been **renamed** to
[`brainwires-agent`](https://crates.io/crates/brainwires-agent) (singular).

The framework's naming rule is **singular for capability domains** — the
plural was the outlier, alongside `brainwires-tools`, `brainwires-providers`,
and `brainwires-permissions`, which were singularized at the same time.

There is no re-export shim — depending on this crate gets you nothing.

## Migration

```toml
# Before
brainwires-agents = "0.10"

# After
brainwires-agent = "0.11"
```

```rust
// Before
use brainwires_agents::{ChatAgent, AgentRuntime, TaskManager, AgentRole};

// After
use brainwires_agent::{ChatAgent, AgentRuntime, TaskManager, AgentRole};
```

The public API is otherwise unchanged. Feature flags are unchanged
(`native`, `wasm`, `eval`, `otel`, `telemetry`, `mdap`, `seal`,
`seal-mdap`, `seal-feedback`, `seal-knowledge`, `skills-registry`,
`skills-signing`).
