# Agent System Architecture

This document describes how child agents are spawned, executed, coordinated, and completed in the Brainwires CLI.

---

## Overview

The agent system enables autonomous, concurrent execution of coding tasks. An external MCP client (e.g., Claude Desktop) spawns agents via `agent_spawn`. Each agent gets its own Tokio task, maintains its own conversation with an AI provider, and coordinates with sibling agents through shared infrastructure: a `CommunicationHub` for messaging and a `FileLockManager` for file access control.

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| TaskAgent | `src/agents/task_agent.rs` | Executes individual tasks autonomously |
| AgentPool | `src/agents/pool.rs` | Manages concurrent agent lifecycles |
| CommunicationHub | `src/agents/communication.rs` ¹ | Inter-agent messaging bus |
| FileLockManager | `src/agents/file_locks.rs` ¹ | File access coordination |
| Validation Loop | `src/agents/validation_loop.rs` ¹ | Quality checks before task completion |
| MCP Handler | `src/mcp_server/handler.rs` | JSON-RPC interface for agent management |
| Agent Tools | `src/mcp_server/agent_tools.rs` | MCP tool definitions for spawning/querying |
| Task Type | `src/types/agent.rs` | Task struct and lifecycle methods |

> ¹ Implemented in the `brainwires-agent` framework crate; re-exported into the CLI via
> `pub use brainwires::agents::*` in `src/agents/mod.rs`. Fully accessible from CLI code
> but not present as local files under `src/agents/`.

---

## End-to-End Execution Flow

```
MCP Client sends agent_spawn
  └─> McpServerHandler::handle_agent_tool_call()     [handler.rs:385]
      └─> spawn_agent_impl(args)                      [handler.rs:432]
          ├─> Create Task from description
          ├─> Build AgentContext (working dir, tools, capabilities)
          ├─> Configure TaskAgentConfig (iterations, validation, MDAP)
          ├─> Instantiate TaskAgent
          ├─> tokio::spawn() → TaskAgent::execute()   [handler.rs:556]
          ├─> Store as AgentEntry in HashMap           [handler.rs:579]
          └─> Return agent_id immediately (non-blocking)

TaskAgent::execute() runs on background Tokio task    [task_agent.rs:198]
  ├─> Register with CommunicationHub
  ├─> Set status → Working
  ├─> Main loop (1..max_iterations):
  │   ├─> Call AI provider                            [task_agent.rs:332]
  │   ├─> Check finish_reason                         [task_agent.rs:335]
  │   │   └─> If "end_turn" or "stop":
  │   │       ├─> Run validation loop
  │   │       ├─> If failed: inject feedback, continue
  │   │       └─> If passed: complete and return
  │   ├─> Extract tool_use blocks from response       [task_agent.rs:354]
  │   └─> For each tool:
  │       ├─> Determine lock requirement              [task_agent.rs:389]
  │       ├─> Acquire file lock                       [task_agent.rs:398]
  │       ├─> Execute tool via ToolExecutor           [task_agent.rs:413]
  │       ├─> Add result to conversation              [task_agent.rs:420]
  │       └─> Track in WorkingSet                     [task_agent.rs:433]
  │
  ├─> Completion path:
  │   ├─> Mark task as Completed
  │   ├─> Broadcast TaskResult via CommunicationHub
  │   ├─> Unregister from hub
  │   ├─> Release all file locks
  │   └─> Return TaskAgentResult { success: true }
  │
  └─> Failure path (iteration limit exceeded):
      ├─> Mark task as Failed
      ├─> Broadcast TaskResult
      ├─> Unregister and release locks
      └─> Return TaskAgentResult { success: false }

MCP Client retrieves result:
  └─> agent_await (blocks) or agent_status (polls)
```

---

## Key Types

### TaskAgent

Defined in `src/agents/task_agent.rs:98-118`:

```rust
pub struct TaskAgent {
    id: String,
    task: Arc<RwLock<Task>>,
    provider: Arc<dyn Provider>,
    tool_executor: ToolExecutor,
    communication_hub: Arc<CommunicationHub>,
    file_lock_manager: Arc<FileLockManager>,
    status: Arc<RwLock<TaskAgentStatus>>,
    config: TaskAgentConfig,
    context: Arc<RwLock<AgentContext>>,
}
```

#### Status Lifecycle

Defined at `task_agent.rs:21-35`:

```rust
pub enum TaskAgentStatus {
    Idle,                    // Initial state after construction
    Working(String),         // Actively executing (message describes current step)
    WaitingForLock(String),  // Blocked on file lock held by another agent
    Paused(String),          // Suspended
    Completed(String),       // Successfully finished
    Failed(String),          // Execution failed
}
```

State transitions during `execute()`:

```
Idle → Working → WaitingForLock → Working → ... → Completed
                                                 → Failed
```

#### Configuration

Defined at `task_agent.rs:65-82`:

```rust
pub struct TaskAgentConfig {
    pub max_iterations: u32,         // Default: 100
    pub permission_mode: PermissionMode,
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub validation_config: Option<ValidationConfig>,
    pub mdap_config: Option<MdapConfig>,
}
```

#### Result

Defined at `task_agent.rs:50-63`:

```rust
pub struct TaskAgentResult {
    pub agent_id: String,
    pub task_id: String,
    pub success: bool,
    pub summary: String,
    pub iterations: u32,
}
```

### Task

Defined in `src/types/agent.rs:72-117`:

```rust
pub struct Task {
    pub id: String,
    pub description: String,
    pub status: TaskStatus,         // Pending | InProgress | Completed | Failed | Blocked | Skipped
    pub plan_id: Option<String>,
    pub parent_id: Option<String>,  // For subtasks
    pub children: Vec<String>,      // Subtask IDs
    pub depends_on: Vec<String>,    // Dependency IDs
    pub priority: TaskPriority,     // Low(0) | Normal(1) | High(2) | Urgent(3)
    pub assigned_to: Option<String>,
    pub iterations: u32,
    pub summary: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
}
```

Tasks support hierarchical decomposition via `parent_id` / `children` and dependency ordering via `depends_on`. Constructor variants:

- `Task::new(id, description)` - standalone task
- `Task::new_for_plan(id, description, plan_id)` - task linked to a plan
- `Task::new_subtask(id, description, parent_id)` - child of another task

### AgentContext

Defined at `src/types/agent.rs:9-25`:

```rust
pub struct AgentContext {
    pub working_directory: String,
    pub conversation_history: Vec<Message>,
    pub tools: Vec<Tool>,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, String>,
    pub working_set: WorkingSet,
    pub capabilities: AgentCapabilities,
}
```

---

## Agent Pool

Defined in `src/agents/pool.rs:27-39`:

```rust
pub struct AgentPool {
    max_agents: usize,
    agents: Arc<RwLock<HashMap<String, AgentHandle>>>,
    communication_hub: Arc<CommunicationHub>,
    file_lock_manager: Arc<FileLockManager>,
    provider: Arc<dyn Provider>,
}
```

The pool provides a higher-level API over individual agent management:

| Method | Line | Description |
|--------|------|-------------|
| `spawn_agent(task, context, config)` | 61 | Non-blocking spawn, returns agent_id |
| `get_status(agent_id)` | 105 | Poll current status |
| `await_completion(agent_id)` | 139 | Block until agent finishes |
| `await_all()` | 217 | Block until all agents finish |
| `stop_agent(agent_id)` | 125 | Abort agent, release locks |
| `list_active()` | 157 | List agents with their statuses |
| `cleanup_completed()` | 185 | Collect results from finished agents |
| `shutdown()` | 236 | Stop all agents, drain pool |
| `stats()` | 245 | Return `AgentPoolStats` |

Each agent is stored as an `AgentHandle` containing the `Arc<TaskAgent>` and its `JoinHandle`.

---

## Communication Hub

Defined in `src/agents/communication.rs:472-476`:

```rust
pub struct CommunicationHub {
    channels: Arc<RwLock<HashMap<String, AgentChannel>>>,
    broadcast_channel: AgentChannel,
}
```

Each agent gets a dedicated `AgentChannel` (unbounded mpsc) upon registration. Messages are wrapped in `MessageEnvelope` with sender, recipient, payload, and timestamp.

### Key Methods

| Method | Line | Description |
|--------|------|-------------|
| `register_agent(agent_id)` | 488 | Create channel for agent |
| `unregister_agent(agent_id)` | 498 | Remove channel |
| `send_message(from, to, msg)` | 507 | Direct message to specific agent |
| `broadcast(from, msg)` | 523 | Send to all registered agents |
| `receive_message(agent_id)` | 533 | Blocking receive |
| `try_receive_message(agent_id)` | 543 | Non-blocking receive |

### Message Types

The `AgentMessage` enum (`communication.rs:10-303`) has 50+ variants organized into categories:

**Core messages:**
- `TaskRequest`, `TaskResult`, `StatusUpdate`
- `HelpRequest`, `HelpResponse`
- `Broadcast`, `Custom`

**Agent lifecycle:**
- `AgentSpawned`, `AgentProgress`, `AgentCompleted`

**Lock coordination:**
- `LockContention`, `LockAvailable`, `WaitQueuePosition`

**Operation tracking:**
- `OperationStarted`, `OperationCompleted`
- `GitOperationStarted`, `GitOperationCompleted`

**Conflict resolution:**
- `BuildBlocked`, `FileWriteBlocked`, `ConflictResolved`

**Advanced protocols:**
- Saga transactions, contract-net negotiation, market allocation
- Worktree management, optimistic concurrency, validation messages

---

## File Lock Manager

Defined in `src/agents/file_locks.rs:111-120`:

```rust
pub struct FileLockManager {
    locks: RwLock<HashMap<PathBuf, FileLockState>>,
    default_timeout: Option<Duration>,
    waiting: RwLock<HashMap<String, HashSet<PathBuf>>>,
}
```

### Lock Types

```rust
pub enum LockType {
    Read,   // Shared — multiple agents can hold simultaneously
    Write,  // Exclusive — only one agent at a time
}
```

### How Agents Determine Lock Requirements

In `task_agent.rs:651-676`, `get_lock_requirement()` maps tool names to lock types:

| Tools | Lock Type |
|-------|-----------|
| `read_file`, `list_directory`, `search_code` | `Read` (shared) |
| `write_file`, `edit_file`, `delete_file`, `create_directory` | `Write` (exclusive) |
| All other tools | No lock required |

### Deadlock Prevention

The `would_deadlock()` method (`file_locks.rs:311-373`) performs DFS cycle detection on the wait-for graph before allowing an agent to wait on a lock. If acquiring a lock would create a cycle, the request fails immediately rather than blocking.

### Key Methods

| Method | Line | Description |
|--------|------|-------------|
| `acquire_lock(agent_id, path, type)` | 153 | Acquire with default timeout |
| `acquire_with_wait(agent_id, path, type, timeout)` | 257 | Acquire with retry + deadlock check |
| `release_lock(agent_id, path, type)` | 411 | Release a specific lock |
| `release_all_locks(agent_id)` | 481 | Release all locks for an agent |
| `can_acquire(path, agent_id, type)` | 544 | Check without blocking |
| `list_locks()` | 591 | List all held locks |
| `stats()` | 660 | Return `LockStats` |

Locks use RAII via `LockGuard` — dropping the guard automatically releases the lock.

---

## Validation Loop

Before an agent can report success, it must pass validation checks. This prevents false completion reports (Bug #5: agent reports success without creating files).

### Configuration

Defined in `src/agents/validation_loop.rs:48-61`:

```rust
pub struct ValidationConfig {
    pub checks: Vec<ValidationCheck>,
    pub working_directory: String,
    pub max_retries: usize,
    pub enabled: bool,
    pub working_set_files: Vec<String>,
}
```

### Available Checks

```rust
pub enum ValidationCheck {
    NoDuplicates,                                    // Detect duplicate exports/functions/types
    BuildSuccess { build_type: String },              // Run npm/cargo/tsc build
    SyntaxValid,                                      // Basic syntax error detection
    CustomCommand { command: String, args: Vec<String> }, // Run arbitrary command
}
```

### Validation Flow

When `execute()` detects a completion signal (`finish_reason == "end_turn"` or `"stop"`), it calls `attempt_validated_completion()` (`task_agent.rs:500-609`):

1. **File existence check** — every file in the working set must exist on disk
2. **Duplicate detection** — no duplicate exports, functions, interfaces, or types
3. **Syntax validation** — basic syntax error detection
4. **Build validation** — optional, runs project build if `build_type` configured

If any check fails:
- Feedback is formatted via `format_validation_feedback()` and injected into the conversation
- The method returns `Ok(None)`, causing the main loop to continue
- The agent gets another chance to fix the issues

If all checks pass:
- Task is marked `Completed`
- `TaskResult` is broadcast via `CommunicationHub`
- Agent unregisters and releases all locks
- `TaskAgentResult { success: true }` is returned

---

## MCP Server Integration

When running as an MCP server (`--mcp-server`), the CLI exposes agent management tools via JSON-RPC.

### AgentEntry

Each spawned agent is stored in the handler as:

```rust
struct AgentEntry {
    agent: Arc<TaskAgent>,
    handle: tokio::task::JoinHandle<Result<TaskAgentResult>>,
}
```

### Available MCP Tools

Defined in `src/mcp_server/agent_tools.rs:18-153`:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `agent_spawn` | `description` (required), `working_directory`, `max_iterations`, `enable_validation`, `build_type`, `enable_mdap`, `mdap_k`, `mdap_target_success`, `mdap_preset` | Spawn a new agent |
| `agent_list` | none | List all agents with status |
| `agent_status` | `agent_id` (required) | Get status of a specific agent |
| `agent_stop` | `agent_id` (required) | Stop a running agent |
| `agent_await` | `agent_id` (required), `timeout_secs` | Wait for agent completion |
| `agent_pool_stats` | none | Pool-wide statistics |
| `agent_file_locks` | none | List all held file locks |

### Spawn Implementation

`McpServerHandler::spawn_agent()` (`handler.rs:440-594`) performs:

1. Parse MCP arguments (description, working_directory, etc.)
2. Create `Task` object
3. Build `AgentContext` with tools and capabilities
4. Configure `TaskAgentConfig` including optional MDAP
5. Construct `TaskAgent`
6. Spawn on Tokio via `tokio::spawn()`
7. Store `AgentEntry` in `HashMap<String, AgentEntry>`
8. Return `agent_id`

### Await Implementation

`McpServerHandler::await_agent()` (`handler.rs:659-734`) supports:

- Optional timeout via `timeout_secs` parameter
- Returns `TaskAgentResult` on completion
- Handles cancellation and panic cases from the join handle

---

## MDAP Integration

MDAP (Multi-Dimensional Adaptive Planning) provides multi-agent voting for complex tasks.

### When to Use

| Use MDAP | Skip MDAP |
|----------|-----------|
| Complex algorithms (graphs, caches, concurrency) | Simple patterns (CRUD, basic utilities) |
| Problems taking 15+ iterations | Well-defined problems (<10 iterations) |
| High-stakes correctness requirements | Time-sensitive tasks (MDAP adds latency) |

### Configuration via `agent_spawn`

```json
{
  "name": "agent_spawn",
  "arguments": {
    "description": "Implement LRU cache",
    "enable_mdap": true,
    "mdap_preset": "high_reliability"
  }
}
```

Available presets:
- `default` — k=3 agents, 95% target success rate
- `high_reliability` — k=5 agents, 99% target
- `cost_optimized` — k=2 agents, 90% target

The cost multiplier equals k (3x, 5x, or 2x API calls), but verified testing shows an average **2.3x efficiency gain** on complex problems, offsetting the cost.

---

## Working Set Tracking

Agents track files they create or modify in a `WorkingSet` (defined in `src/types/working_set.rs`). This set is:

- Initialized empty in `AgentContext`
- Updated when file operations succeed (`task_agent.rs:433-441`)
- Passed to validation checks to know which files to validate
- Used in the file existence check to catch the "success without creating file" bug

---

## Spawning an Agent Programmatically

```rust
use brainwires_cli::agents::{TaskAgent, TaskAgentConfig, spawn_task_agent};
use brainwires_cli::types::agent::{Task, AgentContext};

let task = Task::new("task-001", "Implement feature X");
let context = AgentContext {
    working_directory: "/project/path".to_string(),
    tools: tool_registry.get_all().to_vec(),
    capabilities: AgentCapabilities::full_access(),
    ..Default::default()
};
let config = TaskAgentConfig {
    max_iterations: 20,
    validation_config: Some(ValidationConfig::default().with_build("typescript")),
    mdap_config: None,
    ..Default::default()
};

let agent = Arc::new(TaskAgent::new(
    "agent-001".to_string(),
    task,
    provider.clone(),
    communication_hub.clone(),
    file_lock_manager.clone(),
    context,
    config,
));

// Non-blocking spawn on Tokio
let handle: JoinHandle<Result<TaskAgentResult>> = spawn_task_agent(agent);

// Wait for result
let result = handle.await??;
println!("Success: {}, Iterations: {}", result.success, result.iterations);
```

---

## Key Bug Fixes Relevant to Agent Execution

### Bug #1: Off-by-One Iteration Count

**Location:** `task_agent.rs:261`

Changed `if iterations > max_iterations` to `if iterations >= max_iterations`, ensuring agents stop at exactly the configured limit.

### Bug #5: Agent Reports Success Without Creating File

**Location:** `validation_loop.rs:123-140`

Added file existence verification to the validation loop. Every file in the working set must exist on disk before completion is allowed. This was the most critical reliability fix.
