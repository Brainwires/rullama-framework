# rullama-inference

[![Crates.io](https://img.shields.io/crates/v/rullama-inference.svg)](https://crates.io/crates/rullama-inference)
[![Documentation](https://docs.rs/rullama-inference/badge.svg)](https://docs.rs/rullama-inference)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/rullama-framework)

LLM-driven workhorses for the rullama agent framework.

## What this crate is

Everything in the framework that drives an LLM call (chat /
completion / structured output) or constructs prompts for one.
Extracted from `rullama-agent` in 0.11 (Phase 11f) to separate the
"what holds agents together" half (`rullama-agent`, the
coordination crate) from the "what makes them think" half (this
crate).

## Modules

- `chat_agent` — streaming chat completion loop with per-user
  session management. The everyday workhorse.
- `task_agent` — autonomous task execution loop with tool dispatch,
  validation, and lifecycle hooks
- `runtime` — `AgentRuntime` + `run_agent_loop` — generic agentic
  loop driver
- `context` — `AgentContext` config + the `AgentLifecycleHooks` trait
  object
- `agent_hooks` — `AgentLifecycleHooks` trait (lifecycle interception
  during a `TaskAgent` run)
- `pool` — `AgentPool` (TaskAgent pool with concurrent spawning +
  monitoring)
- `task_orchestrator` — `TaskOrchestrator` (dependency-aware
  scheduling across a pool of TaskAgents)
- `cycle_orchestrator` — `CycleOrchestrator` (Plan→Work→Judge cycle
  driver)
- `plan_executor` — `PlanExecutorAgent` (executes an
  LLM-generated plan with approval modes)
- `validation_loop` — `run_validation` quality-check gate before agent
  completion
- `validation_agent` — rule-based + LLM-driven validation
- `validator_agent` — LLM judge for ad-hoc validation
- `planner_agent` — LLM-powered dynamic task planning
- `judge_agent` — LLM-powered cycle evaluation
- `summarization` — history compaction via LLM
- `system_prompts` — registry of agent prompt templates
  (`AgentPromptKind`, `build_agent_prompt`, etc.)

## Dependency direction

```
rullama-core
  ↑
rullama-agent       (coordination + patterns + schema)
  ↑
rullama-inference   (this crate — agent runtime + LLM-driven helpers)
```

inference depends on agent for coordination types
(`CommunicationHub`, `FileLockManager`, `ResourceChecker`, etc.).
That's the intended arrow: inference USES coordination.

## Features

| Flag      | Default | Enables                                                 |
|-----------|---------|---------------------------------------------------------|
| `native`  | yes     | filesystem + process — needed by validation, sandbox    |
| `wasm`    | off     | wasm-compatible build (drops native features)           |
| `otel`    | off     | OpenTelemetry span export for agent execution traces    |

## Migration from `rullama-agent`

```toml
# Before
rullama-agent = "0.10"

# After — pull both
rullama-agent = "0.12"      # coordination
rullama-inference = "0.12"  # workhorses

# Or via the umbrella facade (default features include both):
rullama = { version = "0.12", features = ["full"] }
```

```rust
// Before
use rullama_agent::{ChatAgent, TaskAgent, AgentRuntime, AgentPool};
use rullama_agent::system_prompts::AgentPromptKind;

// After
use rullama_inference::{ChatAgent, TaskAgent, AgentRuntime, AgentPool};
use rullama_inference::system_prompts::AgentPromptKind;

// Or via the facade — old paths keep working:
use rullama::agents::{ChatAgent, TaskAgent};
use rullama::inference::AgentRuntime;
```

## License

MIT OR Apache-2.0
