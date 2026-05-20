# Brainwires CLI Architecture

This document describes the high-level architecture of the brainwires-cli application.

## Overview

Brainwires CLI is an AI-powered agentic command-line tool for autonomous coding assistance. It combines multi-agent orchestration, Model Context Protocol (MCP) integration, infinite context memory, and extensive tool execution capabilities.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer (clap)                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐    │
│  │  chat   │ │  auth   │ │ history │ │ attach  │ │    mcp-server       │    │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────────┬──────────┘    │
└───────┼──────────┼──────────┼──────────┼────────────────────┼───────────────┘
        │          │          │          │                    │
        v          v          v          v                    v
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Core                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────────┐   │
│  │   Agent    │ │  Provider  │ │   Tools    │ │     MCP Server         │   │
│  │   Layer    │ │   Layer    │ │   Layer    │ │       Layer            │   │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └───────────┬────────────┘   │
│        │              │              │                    │                │
│        v              v              v                    v                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Storage & Context Layer                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │ LanceDB  │ │ Knowledge│ │ Message  │ │  Config  │               │   │
│  │  │ Storage  │ │  Graph   │ │  Memory  │ │  Store   │               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. CLI Layer (`src/cli/`)

Entry point for all user interactions. Handles:
- Command parsing with `clap`
- Multiple chat modes (interactive, TUI, batch, MCP server)
- Output formatting (full, plain, JSON)
- Session management

**Key files:**
- `mod.rs` - Command definitions
- `chat/` - Chat command implementation
- `attach.rs` - Session attachment
- `history.rs` - Conversation history management

### 2. Agent Layer (`src/agents/`)

Multi-agent orchestration system for complex task decomposition:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Orchestration                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Orchestrator │───>│  TaskAgent   │───>│  TaskAgent   │       │
│  │   (Parent)   │    │  (Worker 1)  │    │  (Worker 2)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         v                   v                   v                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Communication Hub (Pub/Sub)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                   │                   │                │
│         v                   v                   v                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              File Lock Manager (R/W Locks)              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**CLI-local files:**
- `task_agent.rs` - Autonomous task execution agents
- `orchestrator.rs` - Parent agent that spawns and coordinates workers
- `worker.rs` - Worker agent implementation
- `pool.rs` - Agent lifecycle management

**From `brainwires` framework crate** (re-exported via `pub use brainwires::agents::*`):
- `communication` - Message hub for agent coordination
- `file_locks` - Read/write file locking
- `validation_loop` - Pre-completion validation checks
- `access_control` - Capability-based access control
- `saga`, `contract_net`, `optimistic` - Multi-agent coordination protocols

**MDAP System (`src/mdap/`):**
Multi-Dimensional Adaptive Planning for complex tasks:
- Voting mechanism (k=3-7 agents vote on decisions)
- Task decomposition into microagent subtasks
- Presets: default, high_reliability, cost_optimized

### 3. Provider Layer (`src/providers/`)

Unified interface for AI model providers:

```
┌──────────────────────────────────────────────────────────────┐
│                    Provider Trait                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ async fn complete(&self, messages, config) -> Stream │   │
│  │ fn model_info(&self) -> ModelInfo                    │   │
│  │ fn supports_tools(&self) -> bool                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐             │
│         v                  v                  v             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Anthropic   │  │   OpenAI     │  │   Google     │      │
│  │  Provider    │  │   Provider   │  │   Provider   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │             │
│         v                  v                  v             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Ollama    │  │    Groq      │  │   Mistral    │      │
│  │   Provider   │  │   Provider   │  │   Provider   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

**Features:**
- Streaming responses via async streams
- Model capability detection
- Context window management
- Cost tracking per provider

### 4. Tool Layer (`src/tools/`)

Extensible tool execution system:

| Tool Category | Tools | Description |
|--------------|-------|-------------|
| File Operations | `read_file`, `write_file`, `edit_file`, `list_directory` | File system manipulation |
| Shell | `bash` | Command execution |
| Git | `git_status`, `git_commit`, `git_diff` | Version control |
| Web | `web_fetch`, `web_search` | HTTP requests and search |
| Code Search | `query_codebase` | Semantic code search |
| Validation | `check_duplicates`, `verify_build`, `check_syntax` | Code quality checks |

**Key files:**
- `executor.rs` - Tool dispatch and execution
- `registry.rs` - Tool registration
- `error.rs` - Error classification and retry strategies

### 5. MCP Layer (`src/mcp/`, `src/mcp_server/`)

Model Context Protocol implementation:

```
┌──────────────────────────────────────────────────────────────────┐
│                        MCP Client                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Connects to external MCP servers                        │   │
│  │  Uses their tools in agent workflows                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        MCP Server                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Exposes CLI as MCP server (--mcp-server flag)           │   │
│  │  Agent management: spawn, list, status, stop, await      │   │
│  │  File lock tools: pool_stats, file_locks                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 6. Storage Layer (`src/storage/`)

Persistent storage for conversations and embeddings:

```
┌──────────────────────────────────────────────────────────────┐
│                   LanceDB Vector Storage                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Tiered Memory:                                      │    │
│  │  - Hot: Recent messages (in-memory)                  │    │
│  │  - Warm: Session messages (indexed)                  │    │
│  │  - Cold: Archived messages (compressed)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│  ┌─────────────────────────v─────────────────────────────┐  │
│  │  Semantic Search:                                      │  │
│  │  - FastEmbed embeddings (all-MiniLM-L6-v2)            │  │
│  │  - LRU cache for embedding memoization                │  │
│  │  - Query by content similarity                        │  │
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### 7. Knowledge Layer (framework: `brainwires-knowledge` crate, `knowledge` feature)

Entity extraction and context management:

- **Entity Extraction**: Extracts files, functions, types, variables from messages
- **Relationship Graph**: Tracks co-occurrence, containment, dependencies
- **Smart Context Injection**: Retrieves relevant past messages when needed
- **Infinite Context**: Never lose important information from earlier in conversation

### 8. Auth Layer (`src/auth/`)

Authentication and session management:

- Brainwires Studio backend authentication
- Session token storage
- Direct provider API key support
- Secure keyring storage via `keyring` crate

### 9. Prompting & SEAL (framework: `brainwires-knowledge` crate)

Adaptive prompting and self-evolving learning are implemented in the `brainwires-knowledge`
framework crate, which the CLI integrates as a dependency. These systems are not local `src/`
modules.

**Adaptive Prompting** — selects the most effective prompting technique per task using
k-means task clustering, SEAL quality signals, and BKS/PKS preference learning. Integrates
via `OrchestratorAgent`. See `docs/adaptive-prompting/ADAPTIVE_PROMPTING_IMPLEMENTATION.md`.

**SEAL (Self-Evolving Agentic Learning)** — coreference resolution, query core extraction,
entity observation, and pattern promotion into BKS. Integrated through
`SealKnowledgeCoordinator`. See `docs/seal/SEAL_ARCHITECTURE.md`.

### 10. Session Layer (`src/session/`)

PTY-based session persistence for long-running tasks:

```
┌──────────────────────────────────────────────────────────────┐
│                   Session Management                          │
│  ┌────────────────────┐    ┌────────────────────┐           │
│  │   SessionServer    │    │   SessionClient    │           │
│  │   (Background)     │<-->│   (Attach/Detach)  │           │
│  └────────────────────┘    └────────────────────┘           │
│            │                                                 │
│  ┌─────────v──────────────────────────────────────────────┐ │
│  │  Features:                                              │ │
│  │  - Detach/reattach to running sessions                  │ │
│  │  - PTY multiplexing for parallel tasks                  │ │
│  │  - State persistence across disconnects                 │ │
│  │  - Session recovery after crashes                       │ │
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### 11. IPC Layer (`src/ipc.rs`)

Inter-process communication for agent coordination:

- Unix domain sockets for local communication
- Message serialization/deserialization
- Agent metadata exchange
- Session discovery and management

### 12. Approval Layer (`src/approval/`)

Tool approval modal for human-in-the-loop workflows:

- **Auto/Ask/Reject Modes**: Configurable per tool
- **Approval History**: Track and learn from decisions
- **Bulk Approval**: Handle multiple similar requests
- **Timeout Handling**: Default actions for unattended operation

### 13. Self-Improvement Layer (`src/self_improve/`)

Autonomous code quality improvement system. The controller evaluates the codebase on a
schedule and applies improvement strategies:

**Strategies** (`src/self_improve/strategies/`):
- `clippy.rs` — Runs `cargo clippy` and applies lint fixes
- `dead_code.rs` — Detects and removes unused code
- `doc_gaps.rs` — Identifies public APIs missing rustdoc
- `refactoring.rs` — Suggests structural improvements
- `test_coverage.rs` — Detects untested code paths
- `todo_scanner.rs` — Tracks TODO/FIXME comments

**Core** (`src/self_improve/`):
- `controller.rs` — Schedules and runs strategies
- `feedback_loop.rs` — Learns from outcomes to prioritize strategies
- `metrics.rs` — Tracks improvement statistics
- `safety.rs` — Validates that changes don't break the build
- `task_generator.rs` — Converts strategy findings into agent tasks

### 14. Agent Lifecycle Layer (`src/agent/`)

Background agent process management (distinct from `src/agents/` orchestration):

- `process.rs` — Agent process spawning and monitoring
- `spawn.rs` — Spawn configuration and startup
- `state.rs` — Agent state machine (idle, running, hibernating)
- `hibernate.rs` — Suspending and restoring agent state
- `plan_mode.rs` — Plan-mode integration for agents
- `message_queue.rs` — Persistent message queue for background agents

### 15. Remote Layer (`src/remote.rs`, framework: `brainwires-network` crate, `remote-transport` feature)

Remote relay connector for external orchestration:

```
┌──────────────────────────────────────────────────────────────┐
│                   Remote Control Bridge                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  WebSocket Protocol:                                    │  │
│  │  - Bidirectional message streaming                      │  │
│  │  - Heartbeat/keep-alive                                 │  │
│  │  - Reconnection with state sync                         │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HTTP REST API:                                         │  │
│  │  - Status queries                                       │  │
│  │  - Command submission                                   │  │
│  │  - Result retrieval                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Telemetry:                                             │  │
│  │  - Resource usage metrics                               │  │
│  │  - Task progress updates                                │  │
│  │  - Error reporting                                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 16. Commands Layer (`src/commands/`)

Slash command system for quick actions:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/mode` | Switch between modes (chat, code, etc.) |
| `/model` | Change the active model |
| `/project:index` | Index codebase for RAG |
| `/project:query` | Query indexed codebase |
| `/project:stats` | Show RAG statistics |

**Custom Commands**: Users can define custom commands in configuration.

### 17. TUI Layer (`src/tui/`)

Terminal user interface using `ratatui`:

```
┌──────────────────────────────────────────────────────────────┐
│                     TUI Architecture                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  App State (src/tui/app/):                              │  │
│  │  - Conversation history                                 │  │
│  │  - Tool execution status                                │  │
│  │  - Input handling                                       │  │
│  │  - Modal dialogs                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Event Handlers (src/tui/app/events/):                  │  │
│  │  - Core event dispatch                                  │  │
│  │  - Viewer handlers (console, shell, fullscreen)         │  │
│  │  - Picker handlers (session, tool, file)                │  │
│  │  - Dialog handlers (help, suspend, exit, approval)      │  │
│  │  - Modal handlers (nano editor, git SCM)                │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Components:                                            │  │
│  │  - File explorer with tree view                         │  │
│  │  - Built-in nano-style editor                           │  │
│  │  - Git SCM panel                                        │  │
│  │  - Find/replace dialog                                  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

### Chat Message Flow

```
User Input
    │
    v
┌──────────────┐
│  CLI Parser  │
└──────┬───────┘
       │
       v
┌──────────────┐    ┌──────────────┐
│   Context    │───>│   Provider   │
│   Builder    │    │   (Stream)   │
└──────────────┘    └──────┬───────┘
                          │
                          v
                   ┌──────────────┐
                   │ Tool Calls?  │
                   └──────┬───────┘
                          │
           ┌──────────────┴──────────────┐
           v                             v
    ┌──────────────┐             ┌──────────────┐
    │     No       │             │     Yes      │
    │  Stream out  │             │ Execute Tool │
    └──────────────┘             └──────┬───────┘
                                       │
                                       v
                                ┌──────────────┐
                                │ Tool Result  │
                                │  to Context  │
                                └──────┬───────┘
                                       │
                                       └──> Loop back to Provider
```

### Agent Spawning Flow

```
MCP Client Request
    │
    v
┌──────────────────┐
│  MCP Server      │
│  (agent_spawn)   │
└────────┬─────────┘
         │
         v
┌──────────────────┐
│  Create TaskAgent│
│  with Config     │
└────────┬─────────┘
         │
         v
┌──────────────────┐
│  Register with   │
│  Agent Pool      │
└────────┬─────────┘
         │
         v
┌──────────────────┐    ┌──────────────────┐
│  Agent Execute   │───>│  Tool Execution  │
│  Loop            │<───│  with Locks      │
└────────┬─────────┘    └──────────────────┘
         │
         v
┌──────────────────┐
│  Validation Loop │
│  (if enabled)    │
└────────┬─────────┘
         │
         v
┌──────────────────┐
│  Return Result   │
│  to MCP Client   │
└──────────────────┘
```

## Module Dependencies

```
                              ┌─────────────┐
                              │    cli      │
                              └──────┬──────┘
                                     │
       ┌─────────────┬───────────────┼───────────────┬─────────────┐
       v             v               v               v             v
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│   agent    │ │  commands  │ │  providers │ │    tui     │ │   remote   │
└──────┬─────┘ └────────────┘ └──────┬─────┘ └────────────┘ └────────────┘
       │                             │
       v                             v
┌────────────┐               ┌────────────┐
│   agents   │               │   tools    │
└──────┬─────┘               └──────┬─────┘
       │                            │
       ├────────────┬───────────────┤
       │            │               │
       v            v               v
┌────────────┐ ┌────────────┐ ┌────────────┐
│    mdap    │ │self_improve│ │  approval  │
└────────────┘ └────────────┘ └────────────┘
       │                            │
       └────────────┬───────────────┘
                    │
                    v
            ┌────────────┐
            │  storage   │
            └──────┬─────┘
                   │
    ┌──────────────┼──────────────┐
    v              v              v
┌────────┐   ┌────────────┐  ┌────────────┐
│knowledge│  │   config   │  │   session  │
└────────┘   └────────────┘  └────────────┘
                                   │
                                   v
                              ┌────────────┐
                              │    ipc     │
                              └────────────┘
```
> `knowledge` is implemented in the `brainwires-knowledge` framework crate (not a local `src/` module).

### Module Descriptions

| Module | Location | Purpose |
|--------|----------|---------|
| `cli` | `src/cli/` | Entry point, command parsing |
| `commands` | `src/commands/` | Slash command system |
| `providers` | `src/providers/` | AI model provider abstraction |
| `tui` | `src/tui/` | Terminal user interface |
| `remote` | `src/remote.rs` | Remote relay connector |
| `agent` | `src/agent/` | Background agent lifecycle (spawn, hibernate, state) |
| `agents` | `src/agents/` | Multi-agent orchestration |
| `tools` | `src/tools/` | Tool execution system |
| `mdap` | `src/mdap/` | Multi-dimensional adaptive planning |
| `self_improve` | `src/self_improve/` | Autonomous code quality improvement |
| `approval` | `src/approval/` | Human-in-the-loop approval modal |
| `storage` | `src/storage/` | Persistent storage (LanceDB) |
| `config` | `src/config/` | Configuration management |
| `session` | `src/session/` | PTY session persistence |
| `ipc` | `src/ipc.rs` | Inter-process communication |
| `auth` | `src/auth.rs` | Authentication and session tokens |
| `knowledge` | framework crate | Entity extraction, context graphs (`brainwires-knowledge`) |
| `seal` | framework crate | Self-evolving adaptive learning (`brainwires-knowledge`) |
| `prompting` | framework crate | Adaptive prompting techniques (`brainwires-knowledge`) |

## Error Handling

The codebase uses a unified error type (`AppError`) with categorization:

```rust
pub enum AppError {
    // Agent/task errors
    Agent(String),
    Mdap(String),

    // Tool execution
    Tool(String),
    ToolNotFound(String),
    ToolTimeout { tool: String, timeout_secs: u64 },

    // Storage/persistence
    Storage(String),

    // Authentication
    Auth(String),
    AuthRequired(String),

    // Configuration
    Config(String),
    ConfigMissing(String),

    // Network/API
    Provider(String),
    ProviderRateLimit { provider: String, retry_after_secs: u64 },
    Connection(String),
    Timeout(String),

    // Permissions
    PermissionDenied(String),

    // Other
    Internal(String),
    FileNotFound(String),
    Io(String),
    Cancelled,
}
```

Errors support:
- `is_retryable()` - Whether operation can be retried
- `is_auth_error()` - Authentication-related errors
- `retry_after_secs()` - Suggested retry delay for rate limits

## Configuration

Configuration is stored in `~/.brainwires/`:

| File | Purpose |
|------|---------|
| `config.json` | User preferences, default model, permissions |
| `session.json` | Authentication tokens |
| `mcp_servers.json` | Registered MCP servers |

API keys are stored securely in the system keyring.

## Performance Considerations

1. **LRU Cache for Embeddings**: Avoids re-embedding identical messages
2. **Async Streams**: Real-time response streaming without buffering
3. **File Locks**: Minimal critical section duration
4. **Tiered Storage**: Hot/warm/cold memory tiers for efficiency
5. **Parallel Tool Execution**: When tools are independent

## Testing Strategy

- **Unit Tests**: Core logic in each module
- **Property Tests**: Invariants with random inputs (`proptest`)
- **Concurrent Tests**: Multi-agent coordination, lock contention
- **Integration Tests**: End-to-end MCP server workflows

See `tests/` directory for test files.
