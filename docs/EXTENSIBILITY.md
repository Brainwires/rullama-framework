# Extensibility Guide

This guide covers extension points in the Brainwires framework for researchers and plugin authors.

## Extension Points

The framework is trait-based: implement a trait, pass it to the component, done.

### Core Traits (rullama-core)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `Provider` | `name`, `chat`, `stream_chat` (+`max_output_tokens` default) | AI chat completion backend |
| `EmbeddingProvider` | `embed`, `dimension`, `model_name` | Text embedding generation |
| `VectorStore` | `initialize`, `upsert`, `search`, `delete`, `clear`, `count` | Embedding storage/search |
| `OutputParser` | `parse`, `format_instructions` | Structured LLM output parsing |
| `LifecycleHook` | `name`, `on_event` (+`priority`, `filter` defaults) | Framework event interception |
| `StagingBackend` | `stage`, `commit`, `rollback`, `pending_count` | Two-phase file write commits |

### RAG Traits (rullama-knowledge, feature `rag`)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `Chunker` | `chunk_file` | Custom file chunking strategy |
| `SearchScorer` | `fuse` | Hybrid search result fusion |
| `VectorDatabase` | 10 methods (initialize, store, search, etc.) | Full RAG vector DB |
| `RelationsProvider` | `extract_definitions`, `extract_references`, `supports_language`, `precision_level` | Code symbol extraction |

### Agent Traits (rullama-agent)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `AgentRuntime` | 11 methods (call_provider, execute_tool, etc.) | Custom agent execution loop |
| `LockPersistence` | `try_acquire`, `release`, `release_all_for_agent`, `cleanup_stale` | Cross-process lock backend |
| `CompensableOperation` | `execute`, `compensate`, `description` (+`operation_type` default) | Saga step with rollback |
| `EvaluationCase` | `name`, `category`, `run` | Eval scenario |

### Tool Traits (rullama-tool-runtime)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `ToolExecutor` | `execute`, `available_tools` | Custom tool execution backend |
| `ToolPreHook` | `before_execute` | Pre-execution tool gate |

### MDAP Traits (rullama-mdap)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `TaskDecomposer` | `decompose`, `is_minimal`, `strategy` | Task decomposition strategy |
| `MicroagentProvider` | `chat` | LLM adapter for voting loop |
| `RedFlagValidator` | `validate` | Response quality check |
| `ResultComposer` | `compose` | Subtask output composition |

### Fine-tune Traits (rullama-finetune)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `FineTuneProvider` | 9 methods (create_job, get_status, etc.) | Cloud fine-tuning provider |
| `TrainingBackend` | `name`, `available_devices`, `train` | Local training execution (impl lives in `rullama-finetune`) |

### Other Extension Traits

| Trait | Crate | Purpose |
|-------|-------|---------|
| `TextToSpeech` | rullama-hardware | TTS synthesis backend |
| `SpeechToText` | rullama-hardware | STT transcription backend |
| `LanguageExecutor` | rullama-tool-builtins (interpreters) | Sandboxed code execution |
| `Dataset` | rullama-finetune | Training data container |
| `FormatConverter` | rullama-finetune | Training data format conversion |
| `Tokenizer` | rullama-finetune | Token encoding/counting |
| `ApprovalPolicy` | rullama-autonomy | Autonomous operation approval |
| `GitForge` | rullama-autonomy | Git forge API (GitHub, GitLab) |

---

## Quick Recipes

### "I want to add a custom AI provider"

Implement `Provider` from `rullama::core`:

```rust
use rullama::prelude::*;
use async_trait::async_trait;
use futures::stream::BoxStream;

struct MyProvider;

#[async_trait]
impl Provider for MyProvider {
    fn name(&self) -> &str { "my-provider" }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> anyhow::Result<ChatResponse> {
        let last = messages.iter().rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.text())
            .unwrap_or_default();

        Ok(ChatResponse {
            message: Message::assistant(format!("Response to: {}", last)),
            usage: Usage::new(10, 20),
            finish_reason: Some("stop".to_string()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, anyhow::Result<StreamChunk>> {
        Box::pin(async_stream::stream! {
            let resp = self.chat(messages, tools, options).await?;
            yield Ok(StreamChunk::Text(resp.message.text().unwrap_or_default().to_string()));
            yield Ok(StreamChunk::Done);
        })
    }
}
```

See `crates/rullama/examples/custom_provider.rs` for a complete runnable example.

### "I want custom embeddings"

Implement `EmbeddingProvider` from `rullama::core`:

```rust
use rullama::prelude::*;

struct MyEmbedding { dim: usize }

impl EmbeddingProvider for MyEmbedding {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        // Your embedding model here
        Ok(vec![0.0; self.dim])
    }
    fn dimension(&self) -> usize { self.dim }
    fn model_name(&self) -> &str { "my-embedding-v1" }
    // embed_batch has a default impl; override for native batching
}
```

See `crates/rullama/examples/custom_embedding.rs` for a complete example.

### "I want custom RAG chunking"

Implement `Chunker` from `rullama::cognition::rag::indexer`:

```rust
use rullama::cognition::rag::indexer::{Chunker, CodeChunk, FileInfo, ChunkStrategy, CodeChunker};
use std::sync::Arc;

struct SemanticChunker;

impl Chunker for SemanticChunker {
    fn chunk_file(&self, file_info: &FileInfo) -> Vec<CodeChunk> {
        // Your chunking logic (sentence boundaries, ML segmentation, etc.)
        vec![]
    }
}

// Plug into the pipeline:
let strategy = ChunkStrategy::Custom(Arc::new(SemanticChunker));
let chunker = CodeChunker::new(strategy);
```

### "I want custom search scoring"

Implement `SearchScorer` from `rullama::cognition::rag::bm25_search`:

```rust
use rullama::cognition::rag::bm25_search::{SearchScorer, BM25Result};
use std::sync::Arc;

struct CrossEncoderReranker;

impl SearchScorer for CrossEncoderReranker {
    fn fuse(
        &self,
        vector_results: Vec<(String, f32)>,
        bm25_results: Vec<BM25Result>,
        limit: usize,
    ) -> Vec<(String, f32)> {
        // Your fusion/reranking logic
        vector_results.into_iter().take(limit).collect()
    }
}

// Plug into LanceVectorDB:
// let db = LanceVectorDB::with_path("/path").await?
//     .with_scorer(Arc::new(CrossEncoderReranker));
```

See `crates/rullama/examples/rag_custom_pipeline.rs` for a complete example.

### "I want a custom agent loop"

Implement `AgentRuntime` from `rullama::agents`:

```rust
use rullama::agents::{AgentRuntime, AgentExecutionResult, run_agent_loop};
use rullama::agents::{CommunicationHub, FileLockManager, LockType};

// AgentRuntime requires 11 methods:
//   agent_id, max_iterations, call_provider, extract_tool_uses,
//   is_completion, execute_tool, get_lock_requirement,
//   on_provider_response, on_tool_result, on_completion, on_iteration_limit
//
// Then run it:
// let result = run_agent_loop(my_runtime, &hub, &lock_manager).await?;
```

See `crates/rullama/examples/agent_quickstart.rs` for infrastructure setup.

---

## Feature Flags

The facade crate (`rullama`) gates each subsystem behind a feature flag.

### Researcher bundle

```toml
[dependencies]
rullama = { version = "0.11", features = ["researcher"] }
```

This enables: `providers`, `agents`, `storage`, `rag`, `training`, `datasets`.

### Individual features

| Feature | Enables | Transitive Dependencies |
|---------|---------|------------------------|
| `tools` | `rullama-tool-runtime` + `rullama-tool-builtins` | — |
| `agents` | `rullama-agent` | — |
| `inference` | `rullama-inference` | rullama-agent, rullama-call-policy |
| `storage` | `rullama-storage` (with native) | lancedb, arrow, fastembed |
| `memory` | `rullama-stores` | — |
| `tiered` | `rullama-memory` | rullama-stores |
| `mcp` | `rullama-mcp-client` | rmcp |
| `mcp-server-framework` | `rullama-mcp-server` | — |
| `mdap` | `rullama-mdap` | — |
| `prompting` | `rullama-prompting` | linfa-clustering, ndarray |
| `permissions` | `rullama-permission` | — |
| `rag` | `rullama-rag` + `rullama-storage` | lancedb, tantivy, tree-sitter |
| `providers` | `rullama-provider` | reqwest |
| `seal` | `rullama-seal` | — |
| `eval` | `rullama-eval` | — |
| `agent-network` | `rullama-network` | — |
| `skills` | `rullama-skills` | — |
| `audio` | `rullama-hardware/audio` | — |
| `gpio` | `rullama-hardware/gpio` | — |
| `bluetooth` | `rullama-hardware/bluetooth` | — |
| `network-hardware` | `rullama-hardware/network` | — |
| `datasets` | `rullama-finetune/datasets-full` | — |
| `training` | `rullama-finetune` | cloud-only since v0.11 |
| `autonomy` | `rullama-autonomy` | — |
| `brain` | `rullama-knowledge/knowledge` | — |

### Compound features

| Feature | Composition |
|---------|-------------|
| `researcher` | providers + agents + storage + rag + training + datasets |
| `agent-full` | agents + permissions + prompting + tools |
| `learning` | seal + knowledge + seal/knowledge |
| `full` | Everything |
| `rag-full-languages` | rag + tree-sitter language grammars |

### Default features

`default = ["tools", "agents"]` — minimal agent toolkit without heavy native deps.

---

## Architecture for Plugin Authors

### Crate dependency graph (simplified)

```
rullama (facade)
  ├── rullama-core (always)       ← core traits, types, errors
  ├── rullama-tool-runtime        ← ToolExecutor, ToolRegistry, validation, smart router
  ├── rullama-tool-builtins       ← Built-in tool implementations
  ├── rullama-agent               ← AgentRuntime, CommunicationHub, MDAP, SEAL
  ├── rullama-provider            ← Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex AI
  ├── rullama-provider-speech     ← TTS / STT providers
  ├── rullama-knowledge           ← BKS / PKS, BrainClient, entity graph
  ├── rullama-rag                 ← Codebase indexing + hybrid retrieval
  ├── rullama-prompting           ← Adaptive prompting
  ├── rullama-network             ← IPC, remote, mesh, LAN discovery
  ├── rullama-mcp-client          ← MCP client
  ├── rullama-mcp-server          ← MCP server framework
  ├── rullama-storage             ← StorageBackend trait, embeddings, BM25, LanceDB
  ├── rullama-stores              ← Schema + CRUD: sessions, tasks, plans, conversations, …
  ├── rullama-memory              ← TieredMemory orchestration + dream consolidation
  └── rullama-finetune            ← Cloud fine-tune APIs + dataset pipelines
                                     ← (local PEFT moved to rullama-finetune)
```

### Where to define new traits

- **Pure types/traits with no heavy deps** → `rullama-core`
- **Tool framework** → `rullama-tool-runtime` (the `ToolExecutor` trait + dispatch)
- **Concrete tool implementations** → `rullama-tool-builtins`
- **Agent coordination** → `rullama-agent`
- **RAG pipeline components** → `rullama-rag`
- **A new persisted store** → `rullama-stores` (schema + CRUD)
- **Memory orchestration** → `rullama-memory` (engines over the schema stores)

### Error handling

Use `FrameworkError` from `rullama::core` for domain-specific errors:

```rust
use rullama::prelude::*;

// Domain-specific constructors:
FrameworkError::provider_auth("my-provider", "Invalid API key")
FrameworkError::provider_model("my-provider", "gpt-5", "Model not found")
FrameworkError::embedding_dimension(384, 768)
FrameworkError::storage_schema("my-store", "Missing 'embeddings' table")
FrameworkError::training_config("learning_rate", "Must be between 0 and 1")

// Generic fallback (wrap any error):
FrameworkError::Provider("Something went wrong".to_string())
```

### Testing your extension

```bash
# Build with just the features you need
cargo build -p rullama --features providers

# Run examples
cargo run -p rullama --example custom_provider --features providers
cargo run -p rullama --example custom_embedding
cargo run -p rullama --example agent_quickstart --features agents
cargo run -p rullama --example rag_custom_pipeline --features rag
```
