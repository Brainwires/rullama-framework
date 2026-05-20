# Extensibility Guide

This guide covers extension points in the Brainwires framework for researchers and plugin authors.

## Extension Points

The framework is trait-based: implement a trait, pass it to the component, done.

### Core Traits (brainwires-core)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `Provider` | `name`, `chat`, `stream_chat` (+`max_output_tokens` default) | AI chat completion backend |
| `EmbeddingProvider` | `embed`, `dimension`, `model_name` | Text embedding generation |
| `VectorStore` | `initialize`, `upsert`, `search`, `delete`, `clear`, `count` | Embedding storage/search |
| `OutputParser` | `parse`, `format_instructions` | Structured LLM output parsing |
| `LifecycleHook` | `name`, `on_event` (+`priority`, `filter` defaults) | Framework event interception |
| `StagingBackend` | `stage`, `commit`, `rollback`, `pending_count` | Two-phase file write commits |

### RAG Traits (brainwires-knowledge, feature `rag`)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `Chunker` | `chunk_file` | Custom file chunking strategy |
| `SearchScorer` | `fuse` | Hybrid search result fusion |
| `VectorDatabase` | 10 methods (initialize, store, search, etc.) | Full RAG vector DB |
| `RelationsProvider` | `extract_definitions`, `extract_references`, `supports_language`, `precision_level` | Code symbol extraction |

### Agent Traits (brainwires-agent)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `AgentRuntime` | 11 methods (call_provider, execute_tool, etc.) | Custom agent execution loop |
| `LockPersistence` | `try_acquire`, `release`, `release_all_for_agent`, `cleanup_stale` | Cross-process lock backend |
| `CompensableOperation` | `execute`, `compensate`, `description` (+`operation_type` default) | Saga step with rollback |
| `EvaluationCase` | `name`, `category`, `run` | Eval scenario |

### Tool Traits (brainwires-tool-runtime)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `ToolExecutor` | `execute`, `available_tools` | Custom tool execution backend |
| `ToolPreHook` | `before_execute` | Pre-execution tool gate |

### MDAP Traits (brainwires-mdap)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `TaskDecomposer` | `decompose`, `is_minimal`, `strategy` | Task decomposition strategy |
| `MicroagentProvider` | `chat` | LLM adapter for voting loop |
| `RedFlagValidator` | `validate` | Response quality check |
| `ResultComposer` | `compose` | Subtask output composition |

### Fine-tune Traits (brainwires-finetune)

| Trait | Required Methods | Purpose |
|-------|-----------------|---------|
| `FineTuneProvider` | 9 methods (create_job, get_status, etc.) | Cloud fine-tuning provider |
| `TrainingBackend` | `name`, `available_devices`, `train` | Local training execution (impl lives in `rullama-finetune`) |

### Other Extension Traits

| Trait | Crate | Purpose |
|-------|-------|---------|
| `TextToSpeech` | brainwires-hardware | TTS synthesis backend |
| `SpeechToText` | brainwires-hardware | STT transcription backend |
| `LanguageExecutor` | brainwires-tool-builtins (interpreters) | Sandboxed code execution |
| `Dataset` | brainwires-finetune | Training data container |
| `FormatConverter` | brainwires-finetune | Training data format conversion |
| `Tokenizer` | brainwires-finetune | Token encoding/counting |
| `ApprovalPolicy` | brainwires-autonomy | Autonomous operation approval |
| `GitForge` | brainwires-autonomy | Git forge API (GitHub, GitLab) |

---

## Quick Recipes

### "I want to add a custom AI provider"

Implement `Provider` from `brainwires::core`:

```rust
use brainwires::prelude::*;
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

See `crates/brainwires/examples/custom_provider.rs` for a complete runnable example.

### "I want custom embeddings"

Implement `EmbeddingProvider` from `brainwires::core`:

```rust
use brainwires::prelude::*;

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

See `crates/brainwires/examples/custom_embedding.rs` for a complete example.

### "I want custom RAG chunking"

Implement `Chunker` from `brainwires::cognition::rag::indexer`:

```rust
use brainwires::cognition::rag::indexer::{Chunker, CodeChunk, FileInfo, ChunkStrategy, CodeChunker};
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

Implement `SearchScorer` from `brainwires::cognition::rag::bm25_search`:

```rust
use brainwires::cognition::rag::bm25_search::{SearchScorer, BM25Result};
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

See `crates/brainwires/examples/rag_custom_pipeline.rs` for a complete example.

### "I want a custom agent loop"

Implement `AgentRuntime` from `brainwires::agents`:

```rust
use brainwires::agents::{AgentRuntime, AgentExecutionResult, run_agent_loop};
use brainwires::agents::{CommunicationHub, FileLockManager, LockType};

// AgentRuntime requires 11 methods:
//   agent_id, max_iterations, call_provider, extract_tool_uses,
//   is_completion, execute_tool, get_lock_requirement,
//   on_provider_response, on_tool_result, on_completion, on_iteration_limit
//
// Then run it:
// let result = run_agent_loop(my_runtime, &hub, &lock_manager).await?;
```

See `crates/brainwires/examples/agent_quickstart.rs` for infrastructure setup.

---

## Feature Flags

The facade crate (`brainwires`) gates each subsystem behind a feature flag.

### Researcher bundle

```toml
[dependencies]
brainwires = { version = "0.11", features = ["researcher"] }
```

This enables: `providers`, `agents`, `storage`, `rag`, `training`, `datasets`.

### Individual features

| Feature | Enables | Transitive Dependencies |
|---------|---------|------------------------|
| `tools` | `brainwires-tool-runtime` + `brainwires-tool-builtins` | — |
| `agents` | `brainwires-agent` | — |
| `inference` | `brainwires-inference` | brainwires-agent, brainwires-call-policy |
| `storage` | `brainwires-storage` (with native) | lancedb, arrow, fastembed |
| `memory` | `brainwires-stores` | — |
| `tiered` | `brainwires-memory` | brainwires-stores |
| `mcp` | `brainwires-mcp-client` | rmcp |
| `mcp-server-framework` | `brainwires-mcp-server` | — |
| `mdap` | `brainwires-mdap` | — |
| `prompting` | `brainwires-prompting` | linfa-clustering, ndarray |
| `permissions` | `brainwires-permission` | — |
| `rag` | `brainwires-rag` + `brainwires-storage` | lancedb, tantivy, tree-sitter |
| `providers` | `brainwires-provider` | reqwest |
| `seal` | `brainwires-seal` | — |
| `eval` | `brainwires-eval` | — |
| `agent-network` | `brainwires-network` | — |
| `skills` | `brainwires-skills` | — |
| `audio` | `brainwires-hardware/audio` | — |
| `gpio` | `brainwires-hardware/gpio` | — |
| `bluetooth` | `brainwires-hardware/bluetooth` | — |
| `network-hardware` | `brainwires-hardware/network` | — |
| `datasets` | `brainwires-finetune/datasets-full` | — |
| `training` | `brainwires-finetune` | cloud-only since v0.11 |
| `autonomy` | `brainwires-autonomy` | — |
| `brain` | `brainwires-knowledge/knowledge` | — |

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
brainwires (facade)
  ├── brainwires-core (always)       ← core traits, types, errors
  ├── brainwires-tool-runtime        ← ToolExecutor, ToolRegistry, validation, smart router
  ├── brainwires-tool-builtins       ← Built-in tool implementations
  ├── brainwires-agent               ← AgentRuntime, CommunicationHub, MDAP, SEAL
  ├── brainwires-provider            ← Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex AI
  ├── brainwires-provider-speech     ← TTS / STT providers
  ├── brainwires-knowledge           ← BKS / PKS, BrainClient, entity graph
  ├── brainwires-rag                 ← Codebase indexing + hybrid retrieval
  ├── brainwires-prompting           ← Adaptive prompting
  ├── brainwires-network             ← IPC, remote, mesh, LAN discovery
  ├── brainwires-mcp-client          ← MCP client
  ├── brainwires-mcp-server          ← MCP server framework
  ├── brainwires-storage             ← StorageBackend trait, embeddings, BM25, LanceDB
  ├── brainwires-stores              ← Schema + CRUD: sessions, tasks, plans, conversations, …
  ├── brainwires-memory              ← TieredMemory orchestration + dream consolidation
  └── brainwires-finetune            ← Cloud fine-tune APIs + dataset pipelines
                                     ← (local PEFT moved to rullama-finetune)
```

### Where to define new traits

- **Pure types/traits with no heavy deps** → `brainwires-core`
- **Tool framework** → `brainwires-tool-runtime` (the `ToolExecutor` trait + dispatch)
- **Concrete tool implementations** → `brainwires-tool-builtins`
- **Agent coordination** → `brainwires-agent`
- **RAG pipeline components** → `brainwires-rag`
- **A new persisted store** → `brainwires-stores` (schema + CRUD)
- **Memory orchestration** → `brainwires-memory` (engines over the schema stores)

### Error handling

Use `FrameworkError` from `brainwires::core` for domain-specific errors:

```rust
use brainwires::prelude::*;

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
cargo build -p brainwires --features providers

# Run examples
cargo run -p brainwires --example custom_provider --features providers
cargo run -p brainwires --example custom_embedding
cargo run -p brainwires --example agent_quickstart --features agents
cargo run -p brainwires --example rag_custom_pipeline --features rag
```
