# brainwires-inference

[![Crates.io](https://img.shields.io/crates/v/brainwires-inference.svg)](https://crates.io/crates/brainwires-inference)
[![Documentation](https://docs.rs/brainwires-inference/badge.svg)](https://docs.rs/brainwires-inference)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

LLM-driven workhorses for the Brainwires Agent Framework.

## What this crate is

Everything in the framework that drives an LLM call (chat /
completion / structured output) or constructs prompts for one.
Extracted from `brainwires-agent` in 0.11 (Phase 11f) to separate the
"what holds agents together" half (`brainwires-agent`, the
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
brainwires-core
  ↑
brainwires-agent       (coordination + patterns + schema)
  ↑
brainwires-inference   (this crate — agent runtime + LLM-driven helpers)
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

## Migration from `brainwires-agent`

```toml
# Before
brainwires-agent = "0.10"

# After — pull both
brainwires-agent = "0.11"      # coordination
brainwires-inference = "0.11"  # workhorses

# Or via the umbrella facade (default features include both):
brainwires = { version = "0.11", features = ["full"] }
```

```rust
// Before
use brainwires_agent::{ChatAgent, TaskAgent, AgentRuntime, AgentPool};
use brainwires_agent::system_prompts::AgentPromptKind;

// After
use brainwires_inference::{ChatAgent, TaskAgent, AgentRuntime, AgentPool};
use brainwires_inference::system_prompts::AgentPromptKind;

// Or via the facade — old paths keep working:
use brainwires::agents::{ChatAgent, TaskAgent};
use brainwires::inference::AgentRuntime;
```

## License

MIT OR Apache-2.0
