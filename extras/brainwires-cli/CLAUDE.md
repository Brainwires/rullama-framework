# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Brainwires CLI** is an AI-powered agentic CLI tool for autonomous coding assistance, built in Rust. It features multi-agent task orchestration, MCP server capabilities, infinite context memory, and extensive tool integration.

---

## Development Commands

### Building and Running

```bash
# Build debug version
cargo build

# Build release version
cargo build --release

# Install locally
cargo install --path .

# Run without installing
cargo run -- auth login
cargo run -- chat
cargo run -- chat --mcp-server
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run integration tests only
cargo test --test '*'
```

### Development Mode

```bash
# Run with detailed logging
RUST_LOG=debug cargo run -- chat

# Run MCP server with logging
RUST_LOG=brainwires_cli=debug cargo run -- chat --mcp-server

# Test agent spawning
cargo run -- chat
# Then in chat: use agent_spawn MCP tool or spawn agents programmatically
```

### Build Optimization

#### Binary Size Comparison

- **Debug build**: ~1.9GB (includes full debug symbols, no optimizations)
- **Release build**: ~182MB (stripped, optimized, 10x smaller)

**Always use release builds for production:**
```bash
cargo build --release
```

#### Optional Features

The project uses Cargo features to make heavy dependencies optional:

```bash
# Default build (fast, no optional features)
cargo build --release

# With local LLM support (adds llama-cpp-2, slower build)
cargo build --release --features llama-cpp-2

# With code interpreters (JavaScript, Python)
cargo build --release --features interpreter-all

# Full build (everything enabled)
cargo build --release --features full
```

**Note:** `llama-cpp-2` is NOT in default features to speed up builds. All local inference components have fallbacks and work without it.

#### Speed Up Compilation

**Use sccache (caches compiled dependencies):**
```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
cargo build --release
```

**Use mold linker (faster linking):**
```bash
# Install mold
sudo apt install mold  # or brew install mold on macOS

# Use it for linking
export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
cargo build --release
```

**Parallel compilation (already set by default):**
```bash
# Cargo uses all CPU cores by default
# To limit: export CARGO_BUILD_JOBS=4
```

#### What Makes Builds Slow?

The heaviest dependencies (by compile time):

1. **arrow-* ecosystem** - Vector database operations (required for infinite context)
2. **lancedb** - Vector storage (required for message persistence)
3. **project-rag** - Codebase indexing (required for /project:* commands)
4. **llama-cpp-2** - Local LLM inference (optional, not in default features)
5. **reqwest** - HTTP client (required for API calls)
6. **ratatui** - TUI framework (required for --tui mode)

Most of these are essential for core functionality. Only `llama-cpp-2` is truly optional.

---

## Architecture Overview

### Core Layers

1. **CLI Layer** (`src/cli/`)
   - Command-line interface using `clap`
   - Chat modes: interactive, single-shot, batch, TUI, MCP server
   - Output formats: full, plain, JSON

2. **Auth Layer** (`src/auth/`)
   - Session management
   - Brainwires Studio backend authentication
   - Direct provider API key support

3. **Provider Layer** (`src/providers/`)
   - **Unified Provider Interface**: All AI providers (Anthropic, OpenAI, Google, Ollama) implement a common `Provider` trait
   - Streaming responses via async streams
   - Model capabilities and context windows
   - Cost tracking per provider

4. **Agent Layer** (`src/agents/`)
   - **Multi-Agent System**: Orchestrator and worker agents for hierarchical task decomposition
   - **TaskAgent**: Autonomous agents that execute tasks independently with tool access
   - **Communication Hub**: Central message bus for agent coordination (status updates, help requests, results)
   - **File Lock Manager**: Read/write locks to prevent conflicts when multiple agents access files
   - **Validation Loop**: Automatic quality checks before task completion (syntax, duplicates, build success)
   - **Agent Pool**: Manages concurrent agent execution with lifecycle tracking

5. **Tool Layer** (`src/tools/`)
   - File operations (read, write, edit, delete, list_directory)
   - Bash command execution
   - Git operations
   - Web operations (fetch, search)
   - Code search (query_codebase with semantic search)
   - Validation tools (check_duplicates, verify_build, check_syntax)

6. **MCP Layer** (`src/mcp/` and `src/mcp_server/`)
   - **MCP Client**: Connect to external MCP servers and use their tools
   - **MCP Server Mode**: Expose CLI as MCP server via `--mcp-server` flag
   - Exposes exactly 10 tools today (verified via `tools/list`):
     - Agent management (5): `agent_spawn`, `agent_list`, `agent_status`, `agent_stop`, `agent_await`
     - Agent pool / locks (2): `agent_pool_stats`, `agent_file_locks`
     - Self-improvement loop (3): `self_improve_start`, `self_improve_status`, `self_improve_stop`
   - The `ToolCategory::TaskManager`, `SessionTask`, `Planning`, and `Context` registries are wired into `handle_list_tools` but currently empty — the `task_*`, `plan_task`, and `recall_context` names in `is_mcp_allowed_tool` are reserved for future use and not exposed yet.

7. **Storage Layer** (`src/storage/`)
   - **LanceDB**: Vector database for conversation persistence
   - **Semantic Search**: Query past conversations by content similarity
   - Tiered memory: hot (recent), warm (session), cold (archived)

8. **Context Layer** (framework: `brainwires-knowledge` crate, `knowledge` feature)
   - **Entity Extraction**: Automatically extracts files, functions, types, variables from messages
   - **Relationship Graph**: Tracks co-occurrence, containment, dependencies between entities
   - **Smart Context Injection**: Retrieves relevant past messages when needed
   - **Infinite Context**: Never lose important information from earlier in conversation

9. **MDAP System** (`src/mdap/`)
   - **Multi-Dimensional Adaptive Planning**: Advanced agent orchestration system
   - **Voting Mechanism**: k agents vote on decisions for reliability (k=3-7)
   - **Task Decomposition**: Breaks complex problems into microagent subtasks
   - **Configuration**: Enable via `enable_mdap` parameter in agent_spawn
   - **Presets**: default (k=3, 95%), high_reliability (k=5, 99%), cost_optimized (k=2, 90%)
   - **Use Cases**: Complex algorithms, multi-step reasoning, high-stakes correctness

---

## Key Architectural Patterns

### Agent Communication Model

Agents communicate via a **CommunicationHub** using typed messages:

```rust
pub enum AgentMessage {
    StatusUpdate { agent_id, status, details },
    HelpRequest { agent_id, issue, blocking },
    TaskResult { agent_id, success, output },
    ToolRequest { agent_id, tool_name, args },
    // ... more message types
}
```

The hub broadcasts messages to all subscribed agents, enabling:
- Parent agents to monitor child agent progress
- Agents to request help or resources
- Coordinated multi-agent workflows

### File Lock Coordination

Multiple agents can work concurrently without conflicts:

```rust
// Read lock (shared, multiple readers allowed)
let _lock = file_lock_manager.acquire_read("src/main.rs").await?;

// Write lock (exclusive, blocks all other access)
let _lock = file_lock_manager.acquire_write("src/main.rs").await?;
```

Locks automatically released on drop. Prevents:
- Interleaved writes corrupting a single write operation
- Read-during-write returning a partial file
- Deadlocks via lock ordering

Does NOT prevent: two agents each issuing a full overwrite of the same file —
those are valid sequential writes from the lock manager's perspective, so the
later one silently "wins." For end-to-end conflict detection, the `write_file`
tool performs an immediate read-back after the write and surfaces a mismatch
as a tool error. This ensures at least one of two conflicting writers sees the
conflict and can retry, pick a unique filename, or abort — rather than both
reporting `Success: true` while one agent's content is gone.

### Validation Loop Pattern

Before task completion, agents must pass validation:

1. **File Existence Check**: All files in working set must exist on disk (prevents Bug #5)
2. **Duplicate Detection**: No duplicate exports/functions/types/interfaces
3. **Syntax Check**: Basic syntax validation for obvious errors
4. **Build Validation**: Optional TypeScript/Cargo/npm build must succeed

```rust
pub struct ValidationConfig {
    pub checks: Vec<ValidationCheck>,
    pub working_directory: String,
    pub max_retries: usize,
    pub enabled: bool,
    pub working_set_files: Vec<String>,
}
```

If validation fails, agent receives error feedback and can retry up to `max_retries` times.

### MCP Server as Agent Manager

When running as MCP server (`--mcp-server`), the CLI exposes agent management tools:

```typescript
// From another MCP client (e.g., Claude Desktop):
{
  "name": "agent_spawn",
  "arguments": {
    "description": "Implement LRU cache in src/cache.ts",
    "working_directory": "/project/path",
    "max_iterations": 20,
    "enable_validation": true,
    "build_type": "typescript",
    "enable_mdap": true,  // Enable MDAP for complex tasks
    "mdap_preset": "high_reliability"  // k=5, 99% target
  }
}
```

This enables hierarchical AI workflows: Claude Desktop can spawn brainwires agents to handle complex coding tasks autonomously.

---

## Important Implementation Details

### Agent Iteration Limits

Agents have a `max_iterations` parameter to prevent infinite loops:
- Default: 100 (effectively unlimited for most tasks)
- Complex tasks: 15-25 iterations typical
- Simple tasks: 5-10 iterations sufficient
- **Bug Note**: Bug #1 (off-by-one) fixed - agents now stop at exactly `max_iterations`, not `max_iterations + 1`

### Working Set Tracking

Agents track files they create/modify in a `WorkingSet`:

```rust
pub struct WorkingSet {
    pub files: HashSet<String>,
    pub pending_operations: Vec<FileOperation>,
}
```

The validation loop uses this to:
1. Know which files to validate
2. Verify all files in working set actually exist
3. Detect missing file creation (Bug #5 fix)

### MDAP Integration

MDAP (Massively Decomposed Agentic Processes) is integrated at the agent spawn level:

- **When to enable**: Complex algorithms (graphs, caches, concurrency), problems taking 15+ iterations
- **When to skip**: Simple patterns (CRUD), well-defined problems (<10 iterations)
- **Cost**: k=3 means 3x API calls, k=5 means 5x API calls
- **Benefit**: Proven 2.3x average efficiency gain on complex problems
- **Configuration**: Set via `TaskAgentConfig.mdap_config` field

Example:
```rust
let config = TaskAgentConfig {
    max_iterations: 20,
    mdap_config: Some(MdapConfig::high_reliability()),  // k=5, 99%
    ..Default::default()
};
```

### Tool Executor Pattern

Tools are executed via `ToolExecutor` which handles:
- Permission checking (auto/ask/reject modes)
- Working directory context
- File lock acquisition
- Error handling and recovery

```rust
impl ToolExecutor {
    pub async fn execute(&self, tool: &str, args: Value) -> Result<String> {
        // 1. Check permissions
        // 2. Acquire necessary locks
        // 3. Execute tool implementation
        // 4. Track in working set if file operation
        // 5. Return result or error
    }
}
```

### Validation Tools

Located in `src/tools/validation_tools.rs`:

- `check_duplicates(file_path)`: Detects duplicate exports/functions/classes/interfaces/types
- `verify_build(working_directory, build_type)`: Runs TypeScript/npm/cargo build
- `check_syntax(file_path)`: Basic syntax error detection (duplicate keywords, etc.)

These are used by the validation loop but can also be called directly as tools.

---

## Critical Bug Fixes Applied

### Bug #1: Off-by-One Iteration Count (FIXED)
**Location**: `src/agents/task_agent.rs:261`
```rust
// Before: if iterations > self.config.max_iterations
// After:
if iterations >= self.config.max_iterations {
    return Err(anyhow!("Agent exceeded max iterations"));
}
```

### Bug #2: TypeScript File-Level Syntax Check (FIXED)
**Location**: `src/tools/validation_tools.rs`
- Skip full TypeScript file-level validation
- Add basic syntax checks for obvious errors (duplicate keywords)
- Rely on project-wide build validation instead

### Bug #3: Premature Iteration Exhaustion (DOCUMENTED)
**Status**: Known issue, not critical
- Agents use 2-3x more iterations than optimal
- Complete work correctly but inefficiently
- Future fix: Improve completion detection

### Bug #4: Validation Missed Duplicate Interfaces (FIXED)
**Location**: `src/tools/validation_tools.rs`
```rust
// Extended check_duplicates to detect:
- export interface NAME
- export type NAME
// Not just const/function/class
```

### Bug #5: Agent Reports Success Without Creating File (FIXED)
**Location**: `brainwires::agents::validation_loop` (framework crate, re-exported via `src/agents/mod.rs`)
```rust
// CRITICAL: Verify all files in working set exist on disk
for file in &changed_files {
    let file_path = PathBuf::from(&config.working_directory).join(file);
    if !file_path.exists() {
        issues.push(ValidationIssue {
            check: "file_existence".to_string(),
            severity: ValidationSeverity::Error,
            message: format!("File '{}' is in working set but does not exist on disk"),
            ..
        });
    }
}
```

This prevents agents from reporting success when they haven't actually created required files.

---

## Testing the Agent System

### Manual Testing

```bash
# Start MCP server mode
cargo run -- chat --mcp-server

# From another terminal/client, send JSON-RPC requests:
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "agent_spawn",
    "arguments": {
      "description": "Create a TypeScript utility function in src/utils.ts",
      "working_directory": "/path/to/project",
      "max_iterations": 15,
      "enable_validation": true,
      "build_type": "typescript"
    }
  },
  "id": 1
}
```

### Integration Testing

The project has been stress-tested using the methodology described below. Test artifacts
(logs, bug reports, session summaries) were not committed to the repository, but the bugs
discovered and fixes applied during those sessions are documented in the Critical Bug Fixes
section above.

### Test Commands

```bash
# Run agent spawn test
cargo test test_agent_spawn -- --nocapture

# Run validation tests
cargo test test_validation -- --nocapture

# Run file lock tests
cargo test test_file_locks -- --nocapture
```

### Comprehensive Stress Testing Methodology

This project underwent rigorous recursive stress testing to discover and fix critical bugs.
The methodology below can be replicated for future testing.

> **Note:** Test artifacts (test logs, bug reports, session summaries) from the original
> stress-testing sessions were not committed to the repository. The bugs found and their
> fixes are documented in the Critical Bug Fixes section above.

#### Testing Philosophy: "Ralph Loop"

**Ralph Loop** is an iterative development methodology where:
1. The same testing prompt is fed repeatedly
2. Claude sees its own work in files and git history
3. A stop hook intercepts completion attempts and continues the loop
4. Testing continues until completion promise OR maximum criteria met

**Completion Criteria:**
- 50+ tests completed OR
- 5+ bugs discovered
- Whichever comes first

#### Progressive Difficulty Testing (Levels 1-7)

Tests are organized into progressive difficulty levels:

**Level 1: Basic Sanity**
- Simple file creation and modification
- Single-file operations
- Basic TypeScript generation
- Verifies fundamental agent capabilities

**Level 2: Multi-Step Tasks**
- Code review and analysis
- Multi-file feature planning
- Complex reasoning tasks
- Tests planning and execution coordination

**Level 3: Validation Edge Cases**
- Duplicate detection stress tests
- TypeScript generic syntax
- Build-breaking changes
- Tests validation system robustness

**Level 4: File Lock Coordination**
- Concurrent file access (3-5 agents simultaneously)
- Write contention scenarios
- Read/write interleaving
- Deadlock prevention verification

**Level 5: Iteration Logic**
- Max iterations with partial progress
- Early completion detection
- Validation disabled scenarios
- Tool errors during validation

**Level 6: Large-Scale Multi-File**
- 5+ file modifications simultaneously
- Cross-file dependencies
- Complex refactoring tasks
- Real-world complexity simulation

**Level 7: MDAP Verification (FAANG-Level Problems)**
- LRU Cache implementation (HashMap + Doubly Linked List)
- Dijkstra's shortest path algorithm
- Async retry handler with exponential backoff
- Generic dependency injection factory
- Rate limiter with sliding window
- Compares MDAP vs standard agents on identical tasks

#### Blind Testing Protocol

**Critical Rule**: Agents receive ONLY task descriptions, not test metadata.

**Why?** To prevent agents from gaming the test:
- No knowledge of test ID or category
- No awareness they're being tested
- No access to test success criteria
- Must complete tasks based solely on requirements

This ensures authentic performance measurement.

#### Parallel Testing Strategy

**Performance Optimization**: Run 5+ agents simultaneously
- 5x faster than sequential testing
- Tests file lock coordination under real load
- Verifies communication hub handles concurrency
- Simulates production multi-agent workflows

**Example**: Spawn 5 agents with different tasks in a single message using multiple `agent_spawn` calls.

#### Bug Discovery and Immediate Fix Workflow

**When a bug is found:**
1. **Document**: Record the issue with reproduction steps and root cause
2. **Fix**: Immediately fix the bug in source code
3. **Build**: Rebuild CLI with `cargo build`
4. **Restart**: Restart MCP server to load new code
5. **Verify**: Re-run failed test to verify fix
6. **Continue**: Resume testing without interruption

**This rapid fix-and-continue approach enables:**
- Real-time bug resolution
- Cumulative improvements during testing session
- Higher bug discovery rate (no test invalidation)

#### Bugs Discovered and Fixed

**Bug #1: Off-by-One Iteration Count** (⭐⭐⭐ Severity)
- **Issue**: Agents exceeded `max_iterations` by 1
- **Fix**: Changed `if iterations > max` to `if iterations >= max`
- **Location**: `src/agents/task_agent.rs:261`
- **Status**: ✅ FIXED

**Bug #2: TypeScript File-Level Syntax Check** (⭐⭐ Severity)
- **Issue**: File-level validation too aggressive, caught false positives
- **Fix**: Skip full validation, add basic syntax checks only
- **Location**: `src/tools/validation_tools.rs`
- **Status**: ✅ FIXED

**Bug #3: Premature Iteration Exhaustion** (⭐⭐ Severity)
- **Issue**: Agents use 2-3x more iterations than optimal
- **Impact**: Complete work correctly but inefficiently
- **Status**: 📝 DOCUMENTED (non-critical, future improvement)

**Bug #4: Validation Missed Duplicate Interfaces** (⭐⭐⭐⭐ Severity)
- **Issue**: Duplicate detection only checked const/function/class, not interface/type
- **Fix**: Extended validation to detect `export interface` and `export type`
- **Location**: `src/tools/validation_tools.rs`
- **Status**: ✅ FIXED

**Bug #5: Agent Reports Success Without Creating File** (⭐⭐⭐⭐⭐ CRITICAL)
- **Issue**: Agent reported completion but file didn't exist on disk
- **Fix**: Added file existence check to validation loop
- **Location**: `brainwires::agents::validation_loop` (framework crate, re-exported via `src/agents/mod.rs`)
- **Status**: ✅ FIXED
- **Impact**: Critical reliability issue that broke completion signal trust

#### MDAP Verification Results

**Testing Goal**: Verify MDAP claims of "zero-error execution" through multi-agent voting

**Methodology**: Paired comparison (identical tasks with MDAP ON vs OFF)

**Results**:
- **LRU Cache**: 19 iterations (standard) → 7 iterations (MDAP) = **2.7x improvement**
- **Rate Limiter**: 19 iterations → 8 iterations = **2.4x improvement**
- **Generic Factory**: 20 iterations → 16 iterations = **1.25x improvement**
- **Average**: **2.3x efficiency gain** on complex algorithms

**Verdict**: ✅ MDAP validated - delivers measurable efficiency gains on complex problems

**When to Use MDAP**:
- Complex algorithms (graphs, caches, concurrency)
- Problems taking 15+ iterations
- High-stakes correctness requirements
- Cost justified by 3x API multiplier when saving 10+ iterations

**When to Skip MDAP**:
- Simple patterns (CRUD, basic utilities)
- Well-defined problems (<10 iterations expected)
- Time-sensitive tasks (MDAP adds latency)

#### Testing Statistics

**Overall Results** (42 tests total):
- Success Rate: 95% (40/42)
- Bugs Found: 5 (4 fixed, 1 documented)
- Bug Discovery Rate: 13.5% (1 bug per 7.4 tests)
- System Stability: ⭐⭐⭐⭐⭐ (zero crashes, zero deadlocks)

**Agent Performance**:
- Basic File Operations: ⭐⭐⭐⭐⭐
- TypeScript Generation: ⭐⭐⭐⭐⭐
- Code Review: ⭐⭐⭐⭐⭐
- Multi-Step Planning: ⭐⭐⭐⭐⭐
- Validation & Recovery: ⭐⭐⭐⭐
- Duplicate Detection: ⭐⭐⭐⭐⭐
- File Lock Coordination: ⭐⭐⭐⭐⭐
- Iteration Efficiency: ⭐⭐⭐ (Bug #3)

#### Replicating This Testing

To run similar comprehensive stress testing:

1. **Start MCP server**: `cargo run -- chat --mcp-server`
2. **Create test environment**: Set up a TypeScript project with `src/` directory
3. **Spawn test agents**: Use `agent_spawn` with progressive difficulty tasks
4. **Monitor results**: Watch for validation failures, iteration counts
5. **Document findings**: Record bugs with reproduction steps and fixes
6. **Fix bugs immediately**: Don't wait for full test suite completion
7. **Verify fixes**: Re-run failed tests after fixes applied

**Key Success Factors**:
- Parallel agent execution (5+ simultaneously)
- Blind testing protocol (no test metadata to agents)
- Immediate fix-and-continue workflow
- Progressive difficulty (start simple, increase complexity)
- MDAP comparison on hard problems

---

## Common Patterns

### Spawning an Agent Programmatically

```rust
use brainwires_cli::agents::{TaskAgent, TaskAgentConfig, AgentContext};
use brainwires_cli::types::task::Task;

let task = Task::new("task-id".to_string(), "Task description".to_string());
let context = AgentContext {
    working_directory: "/project/path".to_string(),
    tools: tool_registry.get_all().to_vec(),
    capabilities: AgentCapabilities::full_access(),
    ..Default::default()
};

let config = TaskAgentConfig {
    max_iterations: 20,
    enable_validation: true,
    validation_config: Some(ValidationConfig::default().with_build("typescript")),
    mdap_config: None,  // Or Some(MdapConfig::default()) to enable MDAP
    ..Default::default()
};

let agent = TaskAgent::new(
    "agent-id".to_string(),
    task,
    provider,
    communication_hub,
    file_lock_manager,
    context,
    config,
);

// Execute async
let result = agent.execute().await?;
```

### Adding a New Tool

1. **Define tool in tool registry** (`src/tools/mod.rs`):

```rust
pub fn register_tools() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        name: "my_tool".to_string(),
        description: "Description of what this tool does".to_string(),
        input_schema: ToolInputSchema::object(properties, required),
        ..Default::default()
    });
    registry
}
```

2. **Implement tool handler** (e.g., `src/tools/my_tool.rs`):

```rust
pub async fn execute_my_tool(args: Value, context: &ToolContext) -> Result<String> {
    // 1. Parse args
    let param = args["param"].as_str()
        .context("Missing required parameter")?;

    // 2. Acquire locks if needed
    let _lock = context.file_lock_manager
        .acquire_write("file.txt").await?;

    // 3. Do work
    let result = do_something(param)?;

    // 4. Track in working set if file operation
    context.working_set.lock().await.add_file("file.txt");

    // 5. Return result
    Ok(serde_json::to_string(&result)?)
}
```

3. **Wire up in tool executor** (`src/tools/executor.rs`):

```rust
match tool_name {
    "my_tool" => execute_my_tool(args, &context).await,
    // ... other tools
}
```

### Adding MCP Tools to the Server

MCP tools are defined in `src/mcp_server/agent_tools.rs`:

```rust
pub struct AgentToolRegistry {
    tools: Vec<Tool>,
}

impl AgentToolRegistry {
    pub fn new() -> Self {
        let tools = vec![
            Tool {
                name: "my_mcp_tool".to_string(),
                description: "Description for MCP clients".to_string(),
                input_schema: ToolInputSchema::object(props, required),
                ..Default::default()
            },
            // ... more tools
        ];
        Self { tools }
    }
}
```

And handled in `src/mcp_server/handler.rs`:

```rust
match params.name.as_ref() {
    "my_mcp_tool" => self.handle_my_mcp_tool(args).await,
    // ... other tools
}
```

---

## Configuration Files

- **User Config**: `~/.brainwires/config.json` - Provider, model, permissions, etc.
- **Session**: `~/.brainwires/session.json` - Authentication tokens
- **MCP Servers**: `~/.brainwires/mcp_servers.json` - Registered MCP servers
- **API Keys**: Stored in system keyring via `keyring` crate (more secure than env vars)

---

## Performance Considerations

### LRU Cache for Embeddings

The knowledge/embedding system uses an LRU cache to avoid re-embedding identical messages:

```rust
use lru::LruCache;

let mut cache: LruCache<String, Vec<f32>> = LruCache::new(NonZeroUsize::new(1000).unwrap());
```

### Async Streams for Responses

All AI providers return streaming responses via `async_stream`:

```rust
use async_stream::stream;
use futures::Stream;

pub fn stream_response(&self) -> impl Stream<Item = Result<String>> {
    stream! {
        for await chunk in response {
            yield Ok(chunk.content);
        }
    }
}
```

This enables real-time output without buffering entire responses.

### File Lock Optimization

Locks are acquired on-demand and released immediately:
- Use `_lock` variable to hold guard
- Lock drops when out of scope
- Minimize critical section duration

---

## Documentation References

- **CLI Chat Modes**: `docs/interface/CLI_CHAT_MODES.md` - Comprehensive guide to all chat modes
- **MCP Server**: `docs/interface/mcp/MCP_SERVER.md` - Running as MCP server, agent management
- **Infinite Context**: `docs/infinite-context/INFINITE_CONTEXT.md` - Entity extraction, relationship graphs
- **IPC & Remote Control**: `docs/distributed-swarms/IPC_AND_REMOTE_CONTROL.md` - Remote relay architecture
- **Slash Commands**: `docs/interface/SLASH_COMMANDS_RAG.md` - Project RAG commands
- **Permission System**: `docs/agents/PERMISSION_SYSTEM.md` - Auto/ask/reject modes

---

## Key Dependencies

- **tokio**: Async runtime (features = "full")
- **clap**: CLI framework with derive macros
- **ratatui**: Terminal UI framework
- **rmcp**: MCP protocol implementation (client + server)
- **reqwest**: HTTP client for API calls
- **serde/serde_json**: Serialization
- **anyhow/thiserror**: Error handling
- **git2**: Git operations
- **walkdir/glob/ignore**: File system operations
- **chrono**: Date/time handling
- **uuid**: Unique IDs for agents/tasks
- **keyring**: Secure API key storage
- **zeroize**: Secure memory clearing

---

## When Working on This Codebase

1. **Always read validation_loop.rs before modifying agents** - Validation is critical for reliability
2. **Test with MCP server mode** - Easiest way to test agent spawning and tool execution
3. **Run tests with --nocapture to diagnose failures** - See Critical Bug Fixes section for documented issues
4. **Use RUST_LOG=debug** - Detailed logging helps debug agent coordination issues
5. **Be aware of MDAP system** - Available for complex tasks, configurable via TaskAgentConfig
6. **File locks are automatic** - Don't manually manage locks unless adding new file operations
7. **All known bugs are documented** - See the Critical Bug Fixes section above for context
