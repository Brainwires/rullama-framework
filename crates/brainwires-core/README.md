# brainwires-core

[![Crates.io](https://img.shields.io/crates/v/brainwires-core.svg)](https://crates.io/crates/brainwires-core)
[![Documentation](https://img.shields.io/docsrs/brainwires-core)](https://docs.rs/brainwires-core)
[![License](https://img.shields.io/crates/l/brainwires-core.svg)](LICENSE)

Core types, traits, and error handling for the Brainwires Agent Framework.

## Overview

`brainwires-core` defines the foundational types shared by every other crate in the framework: messages, tools, providers, tasks, plans, permissions, embeddings, vector stores, and knowledge graphs. It is intentionally lightweight, with no runtime dependencies beyond serialization and async primitives.

**Design principles:**

- **Trait-driven** — `Provider`, `EmbeddingProvider`, `VectorStore`, `StagingBackend` are all trait objects for pluggable implementations
- **Serializable** — every public type implements `Serialize`/`Deserialize` for wire transport and persistence
- **Content-source aware** — trust levels propagate through the system via `ContentSource` tagging
- **WASM-compatible** — the `wasm` feature swaps only the chrono backend, everything else compiles as-is

```text
  ┌──────────────────────────────────────────────────────────────┐
  │                      brainwires-core                         │
  │                                                              │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
  │  │ Message  │  │  Tool    │  │ Provider  │  │   Task    │  │
  │  │ Role     │  │ ToolUse  │  │ (trait)   │  │ TaskStatus│  │
  │  │ Content  │  │ ToolResult│ │ ChatOpts  │  │ Priority  │  │
  │  │ Stream   │  │ Context  │  │ Streaming │  │ Hierarchy │  │
  │  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │
  │                                                              │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
  │  │  Plan    │  │Permission│  │ Embedding │  │  Graph    │  │
  │  │ Metadata │  │  Mode    │  │ (trait)   │  │ Entity    │  │
  │  │ Budget   │  │ RO/Auto/ │  │ Vector    │  │ Edge      │  │
  │  │ Steps    │  │ Full     │  │ Store     │  │ Search    │  │
  │  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │
  │                                                              │
  │  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐   │
  │  │ ContentSource  │  │ WorkingSet   │  │ OutputParser   │   │
  │  │ Trust levels   │  │ LRU eviction │  │ JSON/Regex     │   │
  │  └────────────────┘  └──────────────┘  └────────────────┘   │
  │                                                             │
  │  ┌─────────────────────┐  ┌──────────────────────────────┐  │
  │  │ Event /             │  │ WorkflowCheckpoint /         │  │
  │  │ EventEnvelope<E>    │  │ WorkflowStateStore (trait)   │  │
  │  │ trace_id, sequence  │  │ FsWorkflowStateStore (atomic)│  │
  │  └─────────────────────┘  └──────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-core = "0.11"
```

Build a message, define a tool, and call a provider:

```rust
use brainwires_core::{Message, Tool, ToolInputSchema, ChatOptions, Provider};
use std::collections::HashMap;

let messages = vec![
    Message::system("You are a helpful assistant."),
    Message::user("What is 2 + 2?"),
];

let tool = Tool {
    name: "calculator".into(),
    description: "Evaluate a math expression".into(),
    input_schema: ToolInputSchema::object(
        HashMap::from([("expr".into(), serde_json::json!({"type": "string"}))]),
        vec!["expr".into()],
    ),
    ..Default::default()
};

let options = ChatOptions::deterministic(1024);
let response = provider.chat(&messages, Some(&[tool]), &options).await?;
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `native` | Yes | Native target support, enables `planning` |
| `planning` | Yes (via `native`) | Plan parsing and structured output extraction (`regex`) |
| `wasm` | No | WASM target support (`chrono/wasmbind`) |

```toml
# Default (native + planning)
brainwires-core = "0.11"

# WASM target
brainwires-core = { version = "0.11", default-features = false, features = ["wasm"] }
```

## Architecture

### Messages

The message protocol supports text, images, tool calls, and tool results in a unified structure.

**`Role` variants:** `User`, `Assistant`, `System`, `Tool`

**`MessageContent` variants:**

| Variant | Description |
|---------|-------------|
| `Text(String)` | Simple text content |
| `Blocks(Vec<ContentBlock>)` | Multimodal / structured content |

**`ContentBlock` variants:**

| Variant | Fields | Description |
|---------|--------|-------------|
| `Text` | `text` | Text block |
| `Image` | `source: ImageSource` | Base64-encoded image |
| `ToolUse` | `id`, `name`, `input` | Tool invocation request |
| `ToolResult` | `tool_use_id`, `content`, `is_error` | Tool execution result |

**`Message` constructors:** `Message::user(text)`, `Message::assistant(text)`, `Message::system(text)`, `Message::tool_result(id, content)`.

**`StreamChunk` variants:**

| Variant | Description |
|---------|-------------|
| `Text(String)` | Text delta |
| `ToolUse { id, name }` | Tool use started |
| `ToolInputDelta { id, partial_json }` | Incremental tool input |
| `ToolCall { ... }` | Full tool call request |
| `Usage(Usage)` | Token usage statistics |
| `Done` | Stream completed |

### Provider Trait

The unified AI provider interface supporting both blocking and streaming modes.

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn max_output_tokens(&self) -> Option<u32> { None }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>>;
}
```

**`ChatOptions` presets:**

| Preset | Temperature | Top-P | Use Case |
|--------|-------------|-------|----------|
| `deterministic(max_tokens)` | 0.0 | — | Exact reproduction |
| `factual(max_tokens)` | 0.1 | 0.9 | Factual answers |
| `creative(max_tokens)` | 0.3 | — | Creative tasks |
| `new()` (default) | 0.7 | — | General use |

### Tools

Tool definitions, execution results, idempotency, and staged writes.

**`Tool` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `String` | — | Tool identifier |
| `description` | `String` | — | Human-readable description |
| `input_schema` | `ToolInputSchema` | — | JSON Schema for parameters |
| `requires_approval` | `bool` | `false` | Prompt user before execution |
| `defer_loading` | `bool` | `false` | Lazy-load tool definition |
| `allowed_callers` | `Vec<ToolCaller>` | `[Direct]` | Who can invoke this tool |
| `input_examples` | `Vec<Value>` | `[]` | Example inputs |

**`ToolResult` constructors:** `ToolResult::success(id, content)`, `ToolResult::error(id, error)`.

**`ToolContext`** carries execution environment:

| Field | Type | Description |
|-------|------|-------------|
| `working_directory` | `String` | CWD for file operations |
| `user_id` | `Option<String>` | Current user |
| `metadata` | `HashMap<String, String>` | Arbitrary metadata |
| `capabilities` | `Option<Value>` | Serialized capabilities |
| `idempotency_registry` | `Option<IdempotencyRegistry>` | Deduplicates repeated calls |
| `staging_backend` | `Option<Arc<dyn StagingBackend>>` | Batched writes with rollback |

**`StagingBackend` trait:**

```rust
pub trait StagingBackend: Send + Sync + Debug {
    fn stage(&self, write: StagedWrite) -> bool;
    fn commit(&self) -> Result<CommitResult>;
    fn rollback(&self);
    fn pending_count(&self) -> usize;
}
```

### Tasks

Hierarchical task tracking with status lifecycle and time metrics.

**`TaskStatus` variants:** `Pending`, `InProgress`, `Completed`, `Failed`, `Blocked`, `Skipped`

**`TaskPriority` variants:** `Low` (0), `Normal` (1), `High` (2), `Urgent` (3)

**`Task` key methods:**

| Method | Description |
|--------|-------------|
| `new(id, description)` | Create a standalone task |
| `new_for_plan(id, desc, plan_id)` | Create a plan-linked task |
| `new_subtask(id, desc, parent_id)` | Create a child task |
| `start()` / `complete(summary)` / `fail(error)` | Status transitions |
| `block()` / `skip(reason)` | Blocking and skipping |
| `add_child(id)` / `add_dependency(id)` | Hierarchy management |
| `duration_secs()` / `elapsed_secs()` | Time tracking |

### Plans

Serializable execution plans with budget constraints and step-level granularity.

**`PlanMetadata`** — full plan record with branching support:

| Field | Type | Description |
|-------|------|-------------|
| `plan_id` | `String` | Unique identifier |
| `title` | `String` | Human-readable title |
| `plan_content` | `String` | Full plan text |
| `status` | `PlanStatus` | `Draft` / `Active` / `Paused` / `Completed` / `Abandoned` |
| `parent_plan_id` | `Option<String>` | Branch parent |
| `child_plan_ids` | `Vec<String>` | Sub-plans |
| `depth` | `u32` | Nesting level |
| `embedding` | `Option<Vec<f32>>` | For semantic search |

**`PlanBudget`** — constrains plan execution:

```rust
let budget = PlanBudget::new()
    .with_max_steps(10)
    .with_max_tokens(100_000)
    .with_max_cost_usd(0.50);

budget.check(&plan)?; // Err if plan exceeds any limit
```

### Permissions

Simple three-level permission system.

| Mode | Behavior |
|------|----------|
| `ReadOnly` | Deny all write operations |
| `Auto` (default) | Approve safe ops, prompt for dangerous ones |
| `Full` | Auto-approve everything |

### Content Source

Trust-level classification for injected content, ordered from highest to lowest trust.

| Level | Value | Requires Sanitization |
|-------|-------|----------------------|
| `SystemPrompt` | 0 | No |
| `UserInput` | 1 | No |
| `AgentReasoning` | 2 | No |
| `ExternalContent` | 3 | Yes |

Higher-trust sources can override lower-trust content via `can_override()`.

### Working Set

LRU-managed file context with configurable eviction.

**`WorkingSetConfig` defaults:**

| Field | Default | Description |
|-------|---------|-------------|
| `max_files` | 15 | Maximum tracked files |
| `max_tokens` | 100,000 | Token budget |
| `stale_after_turns` | 10 | Turns before auto-eviction |
| `auto_evict` | `true` | Evict stale entries automatically |

**Key methods:**

| Method | Description |
|--------|-------------|
| `add(path, tokens)` | Add file, returns eviction reason if budget exceeded |
| `add_pinned(path, tokens, label)` | Pinned files cannot be evicted |
| `touch(path)` | Update access count and turn |
| `next_turn()` | Advance turn counter, trigger eviction |
| `total_tokens()` | Current token usage |

**Token estimation:** `estimate_tokens(content)` — approximately 4 characters per token.

### Embedding & Vector Store

Abstract traits for pluggable embedding and storage backends.

**`EmbeddingProvider` trait:**

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}
```

**`VectorStore` trait:**

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn initialize(&self, dimension: usize) -> Result<()>;
    async fn upsert(&self, ids: Vec<String>, embeddings: Vec<Vec<f32>>,
                    contents: Vec<String>, metadata: Vec<Value>) -> Result<usize>;
    async fn search(&self, query: Vec<f32>, limit: usize, min_score: f32)
                    -> Result<Vec<VectorSearchResult>>;
    async fn delete(&self, ids: Vec<String>) -> Result<usize>;
    async fn clear(&self) -> Result<()>;
    async fn count(&self) -> Result<usize>;
}
```

### Knowledge Graph

Entity and relationship types for building code knowledge graphs.

**`EntityType` variants:** `File`, `Function`, `Type`, `Variable`, `Concept`, `Error`, `Command`

**`EdgeType` variants with default weights:**

| Variant | Weight | Description |
|---------|--------|-------------|
| `Defines` | 1.0 | Entity defines another |
| `References` | 0.8 | Entity references another |
| `DependsOn` | 0.7 | Dependency relationship |
| `Modifies` | 0.6 | Entity modifies another |
| `Contains` | 0.5 | Containment relationship |
| `CoOccurs` | 0.3 | Co-occurrence in same context |

**`RelationshipGraphT` trait:**

```rust
pub trait RelationshipGraphT: Send + Sync {
    fn get_node(&self, name: &str) -> Option<&GraphNode>;
    fn get_neighbors(&self, name: &str) -> Vec<&GraphNode>;
    fn get_edges(&self, name: &str) -> Vec<&GraphEdge>;
    fn search(&self, query: &str, limit: usize) -> Vec<&GraphNode>;
    fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>>;
}
```

### Output Parsers (requires `planning` feature)

Structured output extraction from AI text responses.

**`OutputParser` trait:**

```rust
pub trait OutputParser: Send + Sync {
    type Output;
    fn parse(&self, text: &str) -> Result<Self::Output>;
    fn format_instructions(&self) -> String;
}
```

**Built-in parsers:**

| Parser | Output | Description |
|--------|--------|-------------|
| `JsonOutputParser<T>` | `T: DeserializeOwned` | Extracts JSON from markdown fences or prose |
| `JsonListParser<T>` | `Vec<T>` | Extracts JSON arrays |
| `RegexOutputParser` | `HashMap<String, String>` | Extracts named capture groups |

### Error Types

**`FrameworkError` variants:**

| Variant | Description |
|---------|-------------|
| `Config(String)` | Configuration error |
| `Provider(String)` | Provider error |
| `ToolExecution(String)` | Tool execution error |
| `Agent(String)` | Agent error |
| `Storage(String)` | Storage error |
| `PermissionDenied(String)` | Permission denied |
| `Serialization(serde_json::Error)` | JSON serialization error |
| `Other(anyhow::Error)` | Catch-all |

### Workflow State (Crash-Safe Retry)

Persist and resume agent execution across process restarts. Each completed tool call is checkpointed so a restarted agent can skip already-executed side effects.

```rust
use brainwires_core::workflow_state::{
    FsWorkflowStateStore, SideEffectRecord, WorkflowStateStore,
};

let store = FsWorkflowStateStore::with_default_path()?;  // ~/.brainwires/workflow/

// On agent start — check for a prior checkpoint
if let Some(cp) = store.load_checkpoint(&task_id).await? {
    // cp.completed_tool_ids — skip these tool_use_ids
    // cp.step_index — resume iteration count
}

// After each successful tool call
let effect = SideEffectRecord::new(&tool_use_id, "write_file", Some("src/main.rs".into()), true);
store.mark_step_complete(&task_id, &tool_use_id, effect).await?;

// On clean completion
store.delete_checkpoint(&task_id).await?;
```

`FsWorkflowStateStore` writes atomically (write to `.tmp`, then `rename`). Use `InMemoryWorkflowStateStore` in tests.

### Events and Trace IDs

Wrap any domain event with correlation metadata for cross-system tracing without breaking existing types.

```rust
use brainwires_core::event::{EventEnvelope, new_trace_id};

// Generate a trace ID at the start of a logical operation
let trace = new_trace_id();

// Wrap events at boundaries (audit logger, OTel export)
let env = EventEnvelope::new(trace, 0, my_event);
assert_eq!(env.trace_id, trace);

// Map payload while preserving correlation fields
let mapped = env.map(|e| format!("{:?}", e));
assert_eq!(mapped.trace_id, trace);
assert_eq!(mapped.sequence, 0);
```

`TaskAgent` automatically generates a `trace_id` per `execute()` call and stamps it into `ToolContext.metadata["trace_id"]`, enabling correlation with `AuditEvent.metadata` and A2A stream events.

## Usage Examples

### Message Construction

```rust
use brainwires_core::{Message, ContentBlock, MessageContent};

let text_msg = Message::user("Hello!");
let system_msg = Message::system("You are helpful.");
let tool_result = Message::tool_result("call-123", "Result: 42");

// Access text content
if let Some(text) = text_msg.text() {
    println!("{}", text);
}
```

### Tool Definition and Result

```rust
use brainwires_core::{Tool, ToolInputSchema, ToolResult};
use std::collections::HashMap;

let tool = Tool {
    name: "read_file".into(),
    description: "Read a file from disk".into(),
    input_schema: ToolInputSchema::object(
        HashMap::from([("path".into(), serde_json::json!({"type": "string"}))]),
        vec!["path".into()],
    ),
    requires_approval: false,
    ..Default::default()
};

let result = ToolResult::success("call-1", "file contents here");
let error = ToolResult::error("call-2", "file not found");
```

### Chat Options Presets

```rust
use brainwires_core::ChatOptions;

let opts = ChatOptions::deterministic(2048);   // temp=0
let opts = ChatOptions::factual(4096);         // temp=0.1, top_p=0.9
let opts = ChatOptions::creative(4096);        // temp=0.3
let opts = ChatOptions::new().temperature(0.5).max_tokens(8192).system("You are a coder.");
```

### Working Set Management

```rust
use brainwires_core::{WorkingSet, WorkingSetConfig};
use std::path::PathBuf;

let mut ws = WorkingSet::with_config(WorkingSetConfig {
    max_files: 10,
    max_tokens: 50_000,
    stale_after_turns: 5,
    auto_evict: true,
});

ws.add(PathBuf::from("src/main.rs"), 1200);
ws.add_pinned(PathBuf::from("Cargo.toml"), 300, Some("config"));
ws.next_turn();
ws.touch(&PathBuf::from("src/main.rs"));

println!("Files: {}, Tokens: {}", ws.len(), ws.total_tokens());
```

### Plan Budget Validation

```rust
use brainwires_core::{PlanBudget, SerializablePlan, PlanStep};

let plan = SerializablePlan::new(
    "Refactor auth module".into(),
    vec![
        PlanStep { step_number: 1, description: "Read current code".into(), tool_hint: Some("read_file".into()), estimated_tokens: 5000 },
        PlanStep { step_number: 2, description: "Write new impl".into(), tool_hint: Some("write_file".into()), estimated_tokens: 8000 },
    ],
);

let budget = PlanBudget::new()
    .with_max_steps(5)
    .with_max_cost_usd(0.10);

budget.check(&plan)?;
```

### Task Hierarchy

```rust
use brainwires_core::{Task, TaskPriority};

let mut parent = Task::new("task-1", "Refactor module");
parent.set_priority(TaskPriority::High);
parent.start();

let mut child = Task::new_subtask("task-1a", "Update types", "task-1");
child.start();
child.complete("Types updated");

parent.add_child("task-1a".into());
parent.complete("Module refactored");

println!("Duration: {}s", parent.duration_secs().unwrap_or(0));
```

### Content Source Trust

```rust
use brainwires_core::ContentSource;

let source = ContentSource::ExternalContent;
assert!(source.requires_sanitization());

let system = ContentSource::SystemPrompt;
assert!(system.can_override(ContentSource::UserInput));
assert!(!ContentSource::UserInput.can_override(ContentSource::SystemPrompt));
```

## Integration with Brainwires

Use via the `brainwires` facade crate:

```toml
[dependencies]
brainwires = "0.11"
```

Or use standalone — `brainwires-core` has no dependency on any other Brainwires crate.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
