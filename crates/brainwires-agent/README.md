# brainwires-agent

[![Crates.io](https://img.shields.io/crates/v/brainwires-agent.svg)](https://crates.io/crates/brainwires-agent)
[![Documentation](https://img.shields.io/docsrs/brainwires-agent)](https://docs.rs/brainwires-agent)
[![License](https://img.shields.io/crates/l/brainwires-agent.svg)](LICENSE)

Agent orchestration, coordination, and lifecycle management for the Brainwires Agent Framework.

## Overview

`brainwires-agent` provides the multi-agent infrastructure for autonomous task execution. Agents run in a shared pool, communicate through a central hub, coordinate file and resource access via RAII lock guards, and pass through a validation gate before reporting success.

**Design principles:**

- **Async-native** — built on `tokio`, every lock and message operation is non-blocking
- **RAII guards** — file and resource locks are released automatically when guards drop
- **Message-driven** — agents coordinate through a broadcast `CommunicationHub` with 50+ typed message variants
- **Heartbeat liveness** — resource locks are validated against operation heartbeats, not fixed timeouts

```text
                ┌───────────────────────────────────────────────────┐
                │                   AgentPool                       │
                │                                                   │
  spawn ──────► │  TaskAgent ◄──────► CommunicationHub              │
                │      │                     ▲                      │
                │      ▼                     │                      │
                │  ┌─────────┐  ┌────────────┴──────────┐          │
                │  │Execution│  │  AgentMessage (50+)    │          │
                │  │  Graph  │  │  StatusUpdate, Saga,   │          │
                │  └─────────┘  │  ContractNet, Git ...  │          │
                │               └───────────────────────┘          │
                │                                                   │
                │  ┌──────────────┐  ┌───────────────────┐         │
                │  │FileLockMgr   │  │ResourceLockMgr    │         │
                │  │(read/write)  │  │(build/test/git)   │         │
                │  └──────────────┘  └───────────────────┘         │
                │                                                   │
                │  ┌──────────────┐  ┌───────────────────┐         │
                │  │Validation    │  │OperationTracker   │         │
                │  │Loop          │  │(heartbeat liveness)│         │
                │  └──────────────┘  └───────────────────┘         │
                └───────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-agent = "0.11"
```

Spawn a task agent via the pool:

```rust
use std::sync::Arc;
use brainwires_agent::prelude::*;
use brainwires_core::Task;

let hub = Arc::new(CommunicationHub::new());
let locks = Arc::new(FileLockManager::new());

let pool = AgentPool::new(
    10,                      // max concurrent agents
    Arc::clone(&provider),   // AI provider
    Arc::clone(&executor),   // tool executor
    Arc::clone(&hub),
    Arc::clone(&locks),
    "/my/project",
);

let agent_id = pool.spawn_agent(
    Task::new("task-1", "Refactor src/lib.rs"),
    None, // use default TaskAgentConfig
).await?;

let result = pool.await_completion(&agent_id).await?;
println!("{} iterations, success={}", result.iterations, result.success);
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `native` | Yes | Git worktree management (`git2`) and process liveness checking (`libc`) |
| `wasm` | No | WebAssembly-compatible build (disables native-only functionality) |
| `seal` | No | SEAL pipeline: coreference resolution, query extraction, learning, reflection |
| `mdap` | No | MDAP: Multi-Dimensional Adaptive Planning — k-agent voting, microagent decomposition, red flags |
| `seal-mdap` | No | MDAP metric recording for SEAL (enables `seal` + `mdap`) |
| `seal-knowledge` | No | BKS/PKS knowledge system integration for SEAL via `brainwires-knowledge` |
| `seal-feedback` | No | Audit feedback bridge for SEAL via `brainwires-permission` |
| `reasoning` | No | Named reasoning strategies (ReAct, Reflexion, CoT, ToT) and local inference |
| `eval` | No | Evaluation framework (trials, adversarial, regression, stability, ranking metrics — NDCG, MRR, Precision@K) |
| `otel` | No | OpenTelemetry span export for agent execution traces |

Enable features in `Cargo.toml`:

```toml
# Default (native)
brainwires-agent = "0.11"

# WebAssembly target
brainwires-agent = { version = "0.11", default-features = false, features = ["wasm"] }
```

## Architecture

### Agent Lifecycle

A `TaskAgent` runs an AI provider in a loop — calling tools, tracking progress, and validating work before completion.

**Key types:**

- `TaskAgent` — the autonomous execution unit; owns its conversation history and working set
- `TaskAgentConfig` — controls iteration limits, budgets, validation, loop detection, and role
- `TaskAgentResult` — outcome including iterations used, token counts, cost, and failure category
- `FailureCategory` — why an agent stopped (when unsuccessful)
- `ExecutionGraph` — full DAG trace of provider calls and tool invocations
- `RunTelemetry` — aggregate summary derived from the execution graph
- `AgentRole` — least-privilege tool restriction enforced at provider call time

**`FailureCategory` variants:**

| Variant | Description |
|---------|-------------|
| `IterationLimitExceeded` | Exhausted `max_iterations` |
| `TokenBudgetExceeded` | Cumulative tokens exceeded ceiling |
| `CostBudgetExceeded` | Cumulative cost (USD) exceeded ceiling |
| `WallClockTimeout` | `timeout_secs` elapsed |
| `LoopDetected` | Same tool called repeatedly |
| `MaxReplanAttemptsExceeded` | Too many replan cycles |
| `FileScopeViolation` | File operation outside allowed paths |
| `ValidationFailed` | Validation could not be resolved within budget |
| `ToolExecutionError` | Unexpected tool error caused abort |

### Communication

Agents coordinate through a `CommunicationHub` with typed `AgentMessage` variants grouped by protocol.

**Core messages:**

| Group | Variants |
|-------|----------|
| Task lifecycle | `TaskRequest`, `TaskResult`, `StatusUpdate`, `AgentSpawned`, `AgentProgress`, `AgentCompleted` |
| Help & approval | `HelpRequest`, `HelpResponse`, `ApprovalRequest`, `ApprovalResponse`, `Broadcast`, `Custom` |
| Operations | `OperationStarted`, `OperationCompleted`, `LockAvailable`, `WaitQueuePosition`, `LockContention` |
| Git | `GitOperationStarted`, `GitOperationCompleted`, `BuildBlocked`, `FileWriteBlocked`, `ConflictResolved` |
| Saga | `SagaStarted`, `SagaStepCompleted`, `SagaCompleted`, `SagaCompensating` |
| Contract-Net | `TaskAnnounced`, `BidSubmitted`, `TaskAwarded`, `TaskAccepted`, `TaskDeclined` |
| Market | `ResourceAvailable`, `ResourceBidSubmitted`, `ResourceAllocated`, `ResourceReleased` |
| Worktree | `WorktreeCreated`, `WorktreeRemoved`, `WorktreeSwitched` |
| Validation | `ValidationFailed`, `ValidationWarning` |
| Optimistic | `VersionConflict`, `ConflictResolutionApplied` |

**`MessageEnvelope`** wraps every message with `from`, `to`, and `timestamp` metadata.

### File & Resource Locks

Two lock managers handle different coordination layers.

**`FileLockManager`** — path-level read/write locking with deadlock detection:

| Feature | Description |
|---------|-------------|
| Shared reads | Multiple agents can hold `LockType::Read` on the same file |
| Exclusive writes | `LockType::Write` blocks all other access |
| RAII `LockGuard` | Lock released automatically when guard drops |
| Deadlock detection | DFS cycle detection in the wait-for graph |
| Configurable timeout | Per-lock or global default (5 min default) |
| Wait-and-retry | `acquire_with_wait` polls with deadlock checks until timeout |

**`ResourceLockManager`** — operation-level locking for builds, tests, and git:

| `ResourceType` | Conflicts with |
|-----------------|---------------|
| `Build` | `Build`, `BuildTest`, `GitRemoteMerge`, `GitDestructive` |
| `Test` | `Test`, `BuildTest`, `GitRemoteMerge`, `GitDestructive` |
| `BuildTest` | `Build`, `Test`, `BuildTest` |
| `GitIndex` | `GitIndex`, `GitCommit`, `GitRemoteMerge`, `GitDestructive` |
| `GitCommit` | `GitIndex`, `GitCommit`, `GitDestructive` |
| `GitRemoteWrite` | `GitRemoteWrite` |
| `GitRemoteMerge` | `GitRemoteMerge`, `GitIndex`, `Build`, `Test` |
| `GitBranch` | `GitBranch` |
| `GitDestructive` | `GitDestructive`, `GitIndex`, `GitCommit`, `Build`, `Test` |

Resource locks support `ResourceScope::Global` or `ResourceScope::Project(path)` for isolation.

**`AccessControlManager`** — bundles file and resource locks together with contention strategies (`Fail`, `Wait`, `Preempt`).

### Validation

Agents pass through a validation gate before reporting success.

**`ValidationCheck` variants:**

| Check | Description |
|-------|-------------|
| `NoDuplicates` | Detect duplicate exports, functions, types |
| `BuildSuccess { build_type }` | Run `cargo build`, `npm build`, etc. |
| `SyntaxValid` | Basic syntax error detection |
| `CustomCommand { command, args }` | Run an arbitrary validation command |

**`ValidationConfig`** controls which checks run, the working directory, max retries (default 3), and the file working set. `ValidationResult` reports `passed` status and a list of `ValidationIssue` items with severity (`Error` or `Warning`).

### Coordination Patterns

Six coordination patterns are available for multi-agent workflows:

| Pattern | Module | Description |
|---------|--------|-------------|
| **Contract-Net** | `contract_net` | Bidding protocol — manager announces tasks, agents bid, best bidder wins |
| **Saga** | `saga` | Compensating transactions — execute steps in sequence, roll back on failure |
| **Optimistic Concurrency** | `optimistic` | Version-based conflict detection with retry and merge strategies |
| **Wait Queue** | `wait_queue` | FIFO coordination primitive with notification on resource release |
| **Market Allocation** | `market_allocation` | Market-based resource allocation with priority and urgency bidding |
| **Three-State Model** | `state_model` | State snapshots (proposed/committed/rolled-back) for safe rollback |

### Task Management

- **`TaskManager`** — hierarchical task decomposition with dependency tracking
- **`TaskQueue`** — priority-based scheduling with dependency awareness
- **`PlanExecutorAgent`** — executes multi-step plans with configurable approval modes (`Auto`, `StepByStep`, `PlanLevel`)

### Git Coordination

`GitCoordinator` maps git tool operations to their required resource locks:

| Git Tool | Required Locks | Notes |
|----------|---------------|-------|
| `git_status`, `git_diff`, `git_log`, `git_search`, `git_fetch` | None | Read-only |
| `git_stage`, `git_unstage` | `GitIndex` | Modifies staging area |
| `git_commit` | `GitIndex`, `GitCommit` | Creates commit |
| `git_push` | `GitRemoteWrite` | Writes to remote |
| `git_pull` | `GitRemoteMerge`, `GitIndex` | Reads remote, modifies working tree |
| `git_branch` | `GitBranch` | Branch operations |
| `git_discard` | `GitDestructive`, `GitIndex` | Dangerous: loses changes |

### Confidence Scoring

`ResponseConfidence` scores AI responses on a 0.0–1.0 scale based on four factors:

| Factor | Signal |
|--------|--------|
| `completion_confidence` | `finish_reason` — `"stop"` = high, truncated = low |
| `pattern_confidence` | Hedging language ("I think", "possibly") = low |
| `length_confidence` | Response length (normalized) |
| `structure_confidence` | Presence of tool use = higher confidence |

Levels: `very_high` (>= 0.9), `high` (>= 0.8), `medium` (>= 0.6), `low` (>= 0.4), `very_low` (< 0.4).

## Usage Examples

### Spawn Agent via AgentPool

```rust
use std::sync::Arc;
use brainwires_agent::prelude::*;
use brainwires_core::Task;

let pool = AgentPool::new(
    5, provider, executor,
    Arc::new(CommunicationHub::new()),
    Arc::new(FileLockManager::new()),
    "/my/project",
);

let id = pool.spawn_agent(Task::new("t-1", "Add caching layer"), None).await?;
let result = pool.await_completion(&id).await?;
```

### TaskAgentConfig with Validation and Budget

```rust
use brainwires_agent::{TaskAgentConfig, ValidationConfig, ValidationCheck};

let config = TaskAgentConfig {
    max_iterations: 25,
    temperature: 0.2,
    max_tokens: 4096,
    max_total_tokens: Some(500_000),
    max_cost_usd: Some(1.50),
    timeout_secs: Some(300),
    validation_config: Some(ValidationConfig {
        checks: vec![
            ValidationCheck::NoDuplicates,
            ValidationCheck::BuildSuccess { build_type: "cargo".into() },
        ],
        working_directory: "/my/project".into(),
        max_retries: 3,
        enabled: true,
        working_set_files: vec![],
    }),
    ..Default::default()
};
```

### File Lock Acquire / Release

```rust
use std::sync::Arc;
use brainwires_agent::{FileLockManager, LockType};

let manager = Arc::new(FileLockManager::new());

// Shared read — multiple agents can hold simultaneously
let _read_guard = manager.acquire_lock("agent-1", "src/lib.rs", LockType::Read).await?;

// Exclusive write — blocks all other access
let _write_guard = manager.acquire_lock("agent-2", "src/main.rs", LockType::Write).await?;

// Guards release automatically when dropped
```

### CommunicationHub Broadcast

```rust
use std::sync::Arc;
use brainwires_agent::{CommunicationHub, AgentMessage};

let hub = Arc::new(CommunicationHub::new());

hub.register_agent("agent-1".into()).await?;
hub.register_agent("agent-2".into()).await?;

hub.broadcast(
    "orchestrator".into(),
    AgentMessage::Broadcast {
        sender: "orchestrator".into(),
        message: "All agents: pause file writes".into(),
    },
).await?;

let msg = hub.try_receive_message("agent-1").await;
```

### Saga Compensating Transaction

```rust
use brainwires_agent::SagaExecutor;

let mut saga = SagaExecutor::new("agent-1", "edit-and-build");

// Execute steps in sequence
saga.execute_step(Arc::new(file_edit_op)).await?;
saga.execute_step(Arc::new(git_stage_op)).await?;
saga.execute_step(Arc::new(build_op)).await?;

// If any step fails, compensate all completed operations in reverse
if failed {
    let report = saga.compensate_all().await?;
    // Files restored, staging undone
}
```

### Git Coordination

```rust
use std::sync::Arc;
use std::path::PathBuf;
use brainwires_agent::{GitCoordinator, git_tools, get_lock_requirements};

// Check lock requirements for a git operation
let reqs = get_lock_requirements(git_tools::COMMIT);
assert!(!reqs.is_read_only());
assert!(reqs.check_file_conflicts);
assert!(reqs.check_build_conflicts);
```

### Confidence Scoring

```rust
use brainwires_agent::{extract_confidence, ResponseConfidence};

let confidence: ResponseConfidence = extract_confidence(&chat_response);

if confidence.is_high_confidence() {
    println!("Level: {} (score: {:.2})", confidence.level(), confidence.score);
} else {
    let (name, value) = confidence.factors.weakest_factor();
    println!("Low confidence — weakest factor: {} ({:.2})", name, value);
}
```

## Agent Roles

`AgentRole` restricts the tool list passed to the provider at call time — the model never receives tools it cannot use.

| Role | Tool access | Intended use |
|------|-------------|--------------|
| `Exploration` | Read-only: `read_file`, `glob`, `grep`, `search_code`, `web_search`, etc. | Safe exploration of untrusted repos |
| `Planning` | Read + task management: `task_create`, `task_update`, `plan_task`, etc. | Plan-only phase before execution |
| `Verification` | Read + build/test: `execute_command`, `check_duplicates`, `verify_build`, etc. | Post-execution quality checks |
| `Execution` | All tools (default) | Full autonomous execution |

```rust
use brainwires_agent::roles::AgentRole;

let config = TaskAgentConfig {
    role: Some(AgentRole::Exploration),
    ..Default::default()
};
// The agent will only receive read-only tools — write_file, execute_command, etc.
// are filtered before being sent to the provider.
```

Each role also appends a short `[ROLE: ...]` suffix to the system prompt to reinforce constraints.

## System Prompt Registry

`AgentPromptKind` is the authoritative inventory of every agent system prompt in the framework. Use `build_agent_prompt(kind, role)` to construct any prompt — it handles `AgentRole` suffix injection automatically.

```rust
use brainwires_agent::system_prompts::{AgentPromptKind, build_agent_prompt};
use brainwires_agent::roles::AgentRole;

// Default reasoning agent — no role restriction
let prompt = build_agent_prompt(
    AgentPromptKind::Reasoning { agent_id: "agent-1", working_directory: "/project" },
    None,
);

// Exploration role — role suffix appended automatically
let prompt = build_agent_prompt(
    AgentPromptKind::Reasoning { agent_id: "agent-1", working_directory: "/project" },
    Some(AgentRole::Exploration),
);

// MDAP voting microagent — independent reasoning context
let prompt = build_agent_prompt(
    AgentPromptKind::MdapMicroagent {
        agent_id: "micro-2",
        working_directory: "/project",
        vote_round: 2,
        peer_count: 3,
    },
    None,
);
```

**`AgentPromptKind` variants:**

| Variant | Description |
|---------|-------------|
| `Reasoning` | DECIDE → PRE-EVALUATE → EXECUTE → POST-EVALUATE cycle. Default for `TaskAgent`. |
| `Planner` | Read-only exploration; outputs structured JSON task plan. |
| `Judge` | Evaluates Plan→Work cycle results; outputs verdict JSON. |
| `Simple` | Minimal fallback for straightforward tasks. |
| `MdapMicroagent` | One of k independent voting agents; discourages anchoring on peer results. |

To add a new agent type: add a variant to `AgentPromptKind`, implement the function in `system_prompts/agents.rs`, wire it into `build_agent_prompt`.

## Configuration

### TaskAgentConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | `u32` | 100 | Provider call iteration limit |
| `system_prompt` | `Option<String>` | `None` | Override the default reasoning prompt |
| `temperature` | `f32` | 0.0 | AI temperature (0.0–1.0) |
| `max_tokens` | `u32` | 4096 | Max tokens per AI response |
| `validation_config` | `Option<ValidationConfig>` | Some(default) | Quality checks before completion |
| `loop_detection` | `Option<LoopDetectionConfig>` | Some(5-call) | Detect repeated tool calls |
| `goal_revalidation_interval` | `Option<u32>` | Some(10) | Inject goal reminder every N iterations |
| `max_replan_attempts` | `u32` | 3 | Abort after N replan cycles |
| `max_total_tokens` | `Option<u64>` | `None` | Cumulative token ceiling |
| `max_cost_usd` | `Option<f64>` | `None` | Cumulative cost ceiling (USD) |
| `timeout_secs` | `Option<u64>` | `None` | Wall-clock timeout |
| `allowed_files` | `Option<Vec<PathBuf>>` | `None` | File scope whitelist |
| `role` | `Option<AgentRole>` | `None` | Least-privilege tool filter (see `AgentRole`) |

### ValidationConfig Builder

```rust
use brainwires_agent::{ValidationConfig, ValidationCheck};

let config = ValidationConfig {
    checks: vec![
        ValidationCheck::NoDuplicates,
        ValidationCheck::SyntaxValid,
        ValidationCheck::BuildSuccess { build_type: "cargo".into() },
    ],
    working_directory: "/my/project".into(),
    max_retries: 5,
    enabled: true,
    working_set_files: vec!["src/main.rs".into()],
};
```

## Workflow Graph Builder

Build declarative DAG workflows with parallel execution, conditional routing, and shared state:

```rust
use brainwires_agent::workflow::{WorkflowBuilder, WorkflowContext};

let workflow = WorkflowBuilder::new("review-pipeline")
    .node("fetch", |ctx| Box::pin(async move {
        ctx.set("code", serde_json::json!("fn main() {}")).await;
        Ok(serde_json::json!({"status": "fetched"}))
    }))
    .node("lint", |ctx| Box::pin(async move {
        let _code = ctx.get("code").await;
        Ok(serde_json::json!({"lint": "passed"}))
    }))
    .node("review", |ctx| Box::pin(async move {
        Ok(serde_json::json!({"review": "approved"}))
    }))
    .node("summarize", |ctx| Box::pin(async move {
        Ok(serde_json::json!({"summary": "all good"}))
    }))
    .edge("fetch", "lint")
    .edge("fetch", "review")       // lint and review run in parallel
    .edge("lint", "summarize")
    .edge("review", "summarize")   // summarize waits for both
    .build()?;

let result = workflow.run().await?;
assert!(result.success);
```

**Features:**
- Topological validation via `petgraph` (cycle detection)
- Parallel fan-out / fan-in execution
- Conditional routing based on node output
- Shared state via `WorkflowContext` (`Arc<RwLock<HashMap<String, Value>>>`)
- Failure propagation — downstream nodes are skipped when predecessors fail

## Reasoning Strategies

Named reasoning patterns live in the `brainwires-reasoning` crate (they used to be re-exported here under a `reasoning` feature; that compat surface was removed in the pre-1.0 hygiene pass):

```toml
brainwires-reasoning = "0.11"
```

```rust
use brainwires_reasoning::strategies::*;

// Factory creation via preset
let strategy = StrategyPreset::ReAct.create();
println!("{}: {}", strategy.name(), strategy.description());

// Get the system prompt for an agent
let prompt = strategy.system_prompt("agent-1", "/my/project");

// Check completion based on step history
let steps = vec![
    StrategyStep::Thought("Analyzing the problem".into()),
    StrategyStep::Action { tool: "read_file".into(), args: serde_json::json!({"path": "src/lib.rs"}) },
    StrategyStep::Observation("File contents...".into()),
    StrategyStep::FinalAnswer("The function is correct.".into()),
];
assert!(strategy.is_complete(&steps));
```

**Available strategies:**

| Strategy | Pattern | Max Steps |
|----------|---------|-----------|
| `ReActStrategy` | Think → Act → Observe loop | 25 |
| `ReflexionStrategy` | Act → Reflect → Revise loop | 15 |
| `ChainOfThoughtStrategy` | Step-by-step reasoning chain | 10 |
| `TreeOfThoughtsStrategy` | Parallel branch exploration with scoring | 20 |

## OpenTelemetry Export

Export agent execution traces to Jaeger, Datadog, Grafana, or any OpenTelemetry-compatible backend. Requires the `otel` feature:

```toml
brainwires-agent = { version = "0.11", features = ["otel"] }
```

```rust
use brainwires_agent::otel::{export_to_otel, telemetry_attributes};
use opentelemetry::global;

let tracer = global::tracer("brainwires-agent");

// Export full span hierarchy: agent.run → agent.iteration.N → agent.tool.name
export_to_otel(&execution_graph, &telemetry, &tracer);

// Or attach telemetry to an existing span
let attrs = telemetry_attributes(&telemetry);
```

**Span hierarchy:**

```
agent.run (root)
├── agent.iteration.0
│   ├── agent.tool.read_file
│   └── agent.tool.edit_file
├── agent.iteration.1
│   └── agent.tool.bash
└── agent.iteration.2
```

**Span attributes:** `prompt_hash`, `total_iterations`, `total_tool_calls`, `tool_error_count`, `total_prompt_tokens`, `total_completion_tokens`, `total_cost_usd`, `duration_ms`, `success`, `tools_used`

## SEAL — Self-Evolving Agentic Learning (feature: `seal`)

SEAL implements a research-backed framework for enhancing conversational question answering and agent decision-making. Inspired by [Wang et al., arXiv:2512.04868](https://arxiv.org/abs/2512.04868).

```toml
# Core SEAL pipeline
brainwires-agent = { version = "0.11", features = ["seal"] }

# With knowledge system integration
brainwires-agent = { version = "0.11", features = ["seal-knowledge"] }
```

### Pipeline

```text
User Query
    │
    ▼
┌─── Coreference Resolution ─────────────────────────────────────┐
│  detect_references() → resolve() → rewrite_with_resolutions()  │
│  "What uses it?" → "What uses [main.rs]?"                      │
│  Salience: recency(0.35) + frequency(0.15) + centrality(0.20)  │
│            + type_match(0.20) + syntactic(0.10)                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─── Query Core Extraction ──────────────────────────────────────┐
│  classify() → build_expression() → QueryCore                   │
│  S-expression: (JOIN DependsOn ?dep "main.rs")                 │
│  Types: Definition, Location, Dependency, Count, Superlative,  │
│         Enumeration, Boolean, MultiHop                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─── Learning Coordinator ───────────────────────────────────────┐
│  Local Memory (per-session)  │  Global Memory (cross-session)  │
│  Entity tracking, focus      │  Query patterns, tool errors    │
│  Resolution history          │  Resolution patterns, templates │
│  process_query() → match pattern or create new                 │
│  record_outcome() → update reliability scores                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─── Reflection Module ──────────────────────────────────────────┐
│  analyze() → detect issues → suggest fixes → attempt_correction│
│  Errors: EmptyResult, Overflow, EntityNotFound, RelationMismatch│
│  Fixes: RetryWithQuery, ExpandScope, NarrowScope, ResolveEntity│
└────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `SealProcessor` | Main orchestrator chaining all pipeline stages |
| `CoreferenceResolver` | Salience-weighted anaphora resolution ("it" → "[main.rs]") |
| `QueryCoreExtractor` | Natural language → structured S-expression queries |
| `LearningCoordinator` | Dual-level memory (local + global) with pattern learning |
| `ReflectionModule` | Post-execution error detection and automatic correction |
| `SealKnowledgeCoordinator` | BKS/PKS bidirectional bridge (requires `seal-knowledge`) |
| `FeedbackBridge` | Audit log → learning signal converter (requires `seal-feedback`) |

### SealConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_coreference` | `bool` | `true` | Enable coreference resolution stage |
| `enable_query_cores` | `bool` | `true` | Enable query core extraction stage |
| `enable_learning` | `bool` | `true` | Enable self-evolving learning |
| `enable_reflection` | `bool` | `true` | Enable reflection analysis |
| `max_reflection_retries` | `u32` | `2` | Maximum correction attempts per query |
| `min_coreference_confidence` | `f32` | `0.5` | Minimum confidence to accept a resolution |
| `min_pattern_reliability` | `f32` | `0.7` | Minimum pattern reliability for reuse |

### Usage Example

```rust
use brainwires_agent::seal::{SealProcessor, SealConfig, DialogState};
use brainwires_core::graph::{EntityStore, RelationshipGraph};

let mut processor = SealProcessor::with_defaults();
processor.init_conversation("session-001");

let mut dialog = DialogState::default();
dialog.current_turn = 3;
dialog.focus_stack.push("main.rs".to_string());

let entity_store = EntityStore::new();
let graph = RelationshipGraph::new();

// Process a query with an unresolved reference
let result = processor.process(
    "What uses it?",
    &dialog,
    &entity_store,
    Some(&graph),
)?;

println!("Resolved: {}", result.resolved_query);
// → "What uses [main.rs]?"

// Record outcome for learning
processor.record_outcome(
    result.matched_pattern.as_deref(),
    true,   // success
    3,      // result count
    result.query_core.as_ref(),
);

// Inject learned context into future prompts
let context = processor.get_learning_context();
```

### Knowledge Integration (feature: `seal-knowledge`)

The `SealKnowledgeCoordinator` bridges SEAL with the BKS/PKS knowledge system:

| Method | Description |
|--------|-------------|
| `get_pks_context(seal_result)` | Look up personal facts about resolved entities |
| `get_bks_context(query)` | Look up behavioral truths for query context |
| `harmonize_confidence(seal, bks, pks)` | Weighted confidence: SEAL(0.5) + BKS(0.3) + PKS(0.2) |
| `check_and_promote_pattern(pattern, context)` | Promote reliable patterns to BKS |
| `record_tool_failure(tool, error, context)` | Record failure pattern as BKS truth |

Promotion thresholds: >= 80% reliability and >= 5 uses before a pattern is eligible for BKS promotion.

## Integration with Brainwires

Use via the `brainwires` facade crate:

```toml
[dependencies]
brainwires = { version = "0.11", features = ["agents"] }

# With SEAL
brainwires = { version = "0.11", features = ["agents", "seal"] }
```

Or use standalone — `brainwires-agent` depends only on `brainwires-core`, `brainwires-call-policy`, and `brainwires-tool-runtime`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
