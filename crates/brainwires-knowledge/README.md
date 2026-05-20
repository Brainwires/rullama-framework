# brainwires-knowledge

[![Crates.io](https://img.shields.io/crates/v/brainwires-knowledge.svg)](https://crates.io/crates/brainwires-knowledge)
[![Documentation](https://img.shields.io/docsrs/brainwires-knowledge)](https://docs.rs/brainwires-knowledge)
[![License](https://img.shields.io/crates/l/brainwires-knowledge.svg)](LICENSE)

Unified intelligence layer — knowledge graphs, adaptive prompting, RAG, spectral math, and code analysis for the Brainwires Agent Framework.

## Overview

`brainwires-knowledge` consolidates three previously separate crates (`brainwires-brain`, `brainwires-prompting`, `brainwires-rag`) into a single coherent intelligence layer. It provides persistent thought storage with semantic search, adaptive prompting technique selection, codebase indexing with hybrid retrieval, spectral diversity reranking, and AST-aware code analysis.

**Design principles:**

- **Feature-gated composition** — each subsystem activates independently via Cargo features; default builds include only `knowledge` and `prompting`
- **Semantic-first** — all search surfaces (thoughts, code, git history) use vector embeddings for meaning-based retrieval
- **Research-grounded** — prompting techniques from arXiv:2510.18162; spectral reranking from DPP / MSS theory
- **AST-aware** — code chunking and analysis use Tree-sitter parsers for 12 languages, producing structure-preserving chunks

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       brainwires-knowledge                          │
│                                                                     │
│  ┌─── Knowledge (brainwires-brain) ──────────────────────────────┐ │
│  │  BrainClient ──► LanceDB thoughts + semantic search           │ │
│  │  EntityStore / RelationshipGraph ──► entity tracking           │ │
│  │  BKS (behavioral truths) / PKS (personal facts) ──► SQLite    │ │
│  │  FactExtractor ──► automatic categorization + tagging         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─── Prompting (brainwires-prompting) ──────────────────────────┐ │
│  │  TechniqueLibrary ──► 15 techniques in 4 categories           │ │
│  │  TaskClusterManager ──► K-means semantic clustering           │ │
│  │  PromptGenerator ──► multi-source selection (PKS>BKS>cluster) │ │
│  │  LearningCoordinator ──► effectiveness tracking + promotion   │ │
│  │  TemperatureOptimizer ──► adaptive temperature per cluster    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─── RAG (brainwires-rag) ──────────────────────────────────────┐ │
│  │  RagClient ──► index, query, search, git history, navigation  │ │
│  │  Embedding ──► FastEmbed (all-MiniLM-L6-v2, 384d)            │ │
│  │  Indexer ──► FileWalker → CodeChunker → Embedder pipeline     │ │
│  │  Hybrid search ──► vector + BM25 via RRF                     │ │
│  │  Code navigation ──► definitions, references, call graphs     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─── Spectral ──────────┐  ┌─── Code Analysis ─────────────────┐ │
│  │  SpectralReranker     │  │  RepoMap + Relations               │ │
│  │  log-det diversity    │  │  Tree-sitter AST for 12 languages  │ │
│  └───────────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-knowledge = "0.11"
```

Capture a thought and search memory:

```rust
use brainwires_knowledge::knowledge::{BrainClient, CaptureThoughtRequest, SearchMemoryRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BrainClient::new().await?;

    // Store a thought
    client.capture_thought(CaptureThoughtRequest {
        content: "JWT tokens use RS256 signing with 15-minute expiry".into(),
        category: None,
        source: None,
        tags: Some(vec!["auth".into(), "jwt".into()]),
    }).await?;

    // Semantic search
    let results = client.search_memory(SearchMemoryRequest {
        query: "how does authentication work?".into(),
        limit: Some(5),
        min_score: Some(0.7),
        category: None,
    }).await?;

    for thought in &results.results {
        println!("[{:.2}] {}", thought.score, thought.content);
    }

    Ok(())
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `knowledge` | Yes | BrainClient, entity graphs, PKS/BKS, thought capture (LanceDB + SQLite) |
| `prompting` | Yes | 15 prompting techniques, K-means clustering, temperature optimizer |
| `prompting-storage` | No | SQLite persistence for cluster and performance data |
| `spectral` | No | MSS-inspired log-det spectral subset selection for diverse retrieval |
| `rag` | No | Codebase indexing, hybrid search, git history, MCP server binary support |
| `tree-sitter-languages` | No | All 12 Tree-sitter language parsers for AST-aware chunking |
| `code-analysis` | No | Definition/reference lookup and call graph generation |
| `pdf-extract-feature` | No | PDF document extraction |
| `documents` | No | Document processing (zip/docx support) |
| `lancedb-backend` | No | LanceDB vector database backend (forwarded to storage) |
| `qdrant-backend` | No | Qdrant vector database backend (forwarded to storage) |
| `native` | No | Everything: knowledge + prompting + prompting-storage + spectral + rag + code-analysis + documents |
| `wasm` | No | WASM-compatible lightweight build |

```toml
# Default (knowledge + prompting)
brainwires-knowledge = "0.11"

# Full native build
brainwires-knowledge = { version = "0.11", features = ["native"] }

# RAG only
brainwires-knowledge = { version = "0.11", default-features = false, features = ["rag"] }

# WASM target
brainwires-knowledge = { version = "0.11", default-features = false, features = ["wasm"] }
```

## Knowledge Subsystem

*Feature: `knowledge` (default)*

Persistent thought storage, entity graphs, and knowledge systems. Formerly the standalone `brainwires-brain` crate.

### BrainClient

Central API for thought capture and retrieval. Backend-agnostic via `StorageBackend` trait (defaults to LanceDB).

| Method | Description |
|--------|-------------|
| `new()` | Create client with default paths (`~/.brainwires/`) |
| `with_paths(lance, pks, bks)` | Create with custom storage paths |
| `with_backend(backend)` | Create with any `Arc<dyn StorageBackend>` for backend-agnostic storage |
| `capture_thought(req)` | Store a thought with optional category, source, and tags |
| `search_memory(req)` | Semantic vector search across all thoughts |
| `search_knowledge(req)` | Search PKS/BKS knowledge stores |
| `list_recent(req)` | List recent thoughts by timestamp |
| `get_thought(id)` | Get a single thought by ID |
| `memory_stats()` | Get storage statistics (counts, categories, sources) |
| `delete_thought(id)` | Delete a thought by ID |

### Data Model

**`Thought`** — a stored unit of knowledge:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `String` | Unique identifier |
| `content` | `String` | Thought content |
| `category` | `ThoughtCategory` | Classification |
| `source` | `ThoughtSource` | Origin |
| `tags` | `Vec<String>` | Searchable tags |
| `created_at` | `i64` | Unix timestamp |

**`ThoughtCategory`** — `Observation`, `Decision`, `Question`, `Insight`, `Task`, `Reference`, `Other`.

**`ThoughtSource`** — `User`, `Agent`, `System`, `External`, `Conversation`.

### Request/Response Types

| Type | Description |
|------|-------------|
| `CaptureThoughtRequest` / `CaptureThoughtResponse` | Create a thought |
| `SearchMemoryRequest` / `SearchMemoryResponse` | Semantic search with limit, min_score, category filter |
| `SearchKnowledgeRequest` / `SearchKnowledgeResponse` | PKS/BKS knowledge search |
| `ListRecentRequest` / `ListRecentResponse` | Recent thoughts by time |
| `GetThoughtRequest` / `GetThoughtResponse` | Single thought lookup |
| `MemoryStatsRequest` / `MemoryStatsResponse` | Storage statistics |
| `DeleteThoughtRequest` / `DeleteThoughtResponse` | Thought deletion |

### Entity & Relationship Graph

**`EntityStore`** — tracks entities extracted from messages with contradiction detection:

| Method | Description |
|--------|-------------|
| `add_extraction(result, message_id, timestamp)` | Add entities, detect contradictions |
| `get(name, entity_type)` | Lookup entity |
| `get_by_type(entity_type)` | All entities of a type |
| `get_top_entities(limit)` | Most-mentioned entities |
| `get_related(entity_name)` | Related entity names |
| `drain_contradictions()` | Take and clear contradiction events |
| `stats()` | Entity and relationship counts |

**`EntityType`** — `File`, `Function`, `Type`, `Error`, `Concept`, `Variable`, `Command`.

**`Relationship`** — `Defines`, `References`, `Modifies`, `DependsOn`, `Contains`, `CoOccurs`.

**`RelationshipGraph`** — in-memory graph with traversal:

| Method | Description |
|--------|-------------|
| `add_node(name, entity_type)` | Add entity node |
| `add_edge(from, to, edge_type)` | Add relationship edge |
| `get_neighbors(name)` | Adjacent entities |
| `shortest_path(from, to)` | BFS shortest path |
| `importance_score(name)` | Degree centrality score |
| `calculate_importance(entity)` | Compute importance score for an entity (public) |

**`RelationshipGraph::calculate_importance` formula**:

```
score = ln(mention_count).max(0) * 0.3   // mention-count component
      + type_bonus                         // File=0.4, Type=0.35, Function=0.3, Error=0.25, Concept=0.2, Command=0.15, Variable=0.1
      + min(message_spread * 0.05, 0.2)   // recency proxy (capped at 0.2)
```

**Known limitation**: `ln(1) = 0`, so the mention-count component contributes nothing for entities seen exactly once. The type bonus and recency proxy still apply, so the score is always non-zero — but a single-mention entity's score depends solely on its type and message spread. Empirical validation via `brainwires_autonomy::eval::EntitySingleMentionCase` confirms non-zero scoring is maintained.

### Knowledge Systems

- **PKS (Personal Knowledge System)** — user-scoped facts stored in SQLite
- **BKS (Behavioral Knowledge System)** — shared behavioral truths with confidence scoring, promoted from observed patterns
- **FactExtractor** — automatic categorization and tag extraction from free text

### MCP Server

A standalone MCP server binary for the knowledge subsystem is available at `extras/brainwires-brain-server/`.

## Prompting Subsystem

*Feature: `prompting` (default)*

Adaptive prompting technique selection based on arXiv:2510.18162, with BKS/PKS/SEAL integration. Formerly the standalone `brainwires-prompting` crate.

### TechniqueLibrary

15 prompting techniques organized in 4 categories:

| Category | Techniques |
|----------|------------|
| **RoleAssignment** | RolePlaying |
| **EmotionalStimulus** | EmotionPrompting, StressPrompting |
| **Reasoning** | ChainOfThought, LogicOfThought, LeastToMost, ThreadOfThought, PlanAndSolve, SkeletonOfThought, ScratchpadPrompting |
| **Others** | DecomposedPrompting, IgnoreIrrelevantConditions, HighlightedCoT, SkillsInContext, AutomaticInformationFiltering |

Each technique has `TechniqueMetadata` with name, category, description, and `ComplexityLevel` (`Simple`, `Moderate`, `Advanced`) for SEAL quality filtering.

### TaskClusterManager

K-means clustering of tasks by semantic similarity using `linfa-clustering` and `ndarray`. Groups similar tasks so technique effectiveness can be tracked per cluster.

| Method | Description |
|--------|-------------|
| `new(k)` | Create manager with k clusters |
| `fit(embeddings, labels)` | Train clusters on task embeddings |
| `predict(embedding)` | Assign a task to a cluster |
| `cosine_similarity(a, b)` | Utility for vector similarity |

### PromptGenerator

Dynamic prompt generation with multi-source technique selection:

1. **PKS** (Personal Knowledge System) — user-specific preferences, highest priority
2. **BKS** (Behavioral Knowledge System) — proven techniques from observed effectiveness
3. **Cluster default** — fallback based on task cluster membership

```rust
use brainwires_knowledge::prompting::{PromptGenerator, GeneratedPrompt};

let generator = PromptGenerator::new(pks_cache, bks_cache, cluster_manager);
let prompt: GeneratedPrompt = generator.generate(task_embedding, seal_result).await?;
```

### LearningCoordinator

Tracks technique effectiveness per cluster and promotes successful patterns to BKS:

| Method | Description |
|--------|-------------|
| `record_outcome(cluster, technique, success, quality)` | Record technique performance |
| `get_stats(cluster)` | Get technique statistics for a cluster |
| `promote_to_bks()` | Promote high-confidence patterns to BKS |

### TemperatureOptimizer

Adaptive temperature optimization per task cluster based on observed performance:

| Method | Description |
|--------|-------------|
| `optimal_temperature(cluster)` | Get optimized temperature for cluster |
| `record_performance(cluster, temperature, quality)` | Record a temperature/quality observation |

### SealProcessingResult Integration

The prompting system integrates with SEAL quality scores to filter techniques by complexity level — simple techniques for low-quality contexts, advanced techniques for high-quality contexts.

### Storage

Behind the `prompting-storage` feature, `ClusterStorage` provides SQLite persistence for cluster assignments and performance data.

## RAG Subsystem

*Feature: `rag`*

Codebase indexing and semantic search with hybrid retrieval. Formerly the standalone `brainwires-rag` crate.

### RagClient

Core library API for indexing and searching codebases:

| Method | Description |
|--------|-------------|
| `new()` | Create with default configuration |
| `with_config(config)` | Create with custom configuration |
| `with_vector_db(db)` | Create with any `Arc<dyn VectorDatabase>` for backend-agnostic RAG |
| `index_codebase(req)` | Index a directory (full, incremental, or smart mode) |
| `query_codebase(req)` | Semantic code search with hybrid scoring |
| `search_with_filters(req)` | Advanced search with file type, language, and path filters |
| `get_statistics()` | Index statistics (file counts, languages, chunks) |
| `clear_index()` | Delete all indexed data |
| `search_git_history(req)` | Semantic search over git commit history |
| `query_diverse(req, config)` | Query with spectral diversity reranking |
| `find_definition(req)` | Find symbol definition locations (requires `code-analysis`) |
| `find_references(req)` | Find symbol reference locations (requires `code-analysis`) |
| `get_call_graph(req)` | Build call graph for a symbol (requires `code-analysis`) |

### Indexing Pipeline

```text
FileWalker ──► CodeChunker ──► Embedder ──► VectorDB
    │              │
    │              ├── AST-aware (Tree-sitter, 12 languages)
    │              └── Fixed-line fallback
    │
    ├── .gitignore-aware filtering
    ├── Configurable max file size
    └── Incremental updates via hash cache
```

**Supported languages (AST-aware):** Rust, Python, JavaScript, TypeScript, Go, Java, Swift, C, C++, C#, Ruby, PHP.

**Indexing modes:** `Full` (reindex everything), `Incremental` (changed files only), `Smart` (auto-detect).

### Hybrid Search

Search combines two scoring methods via Reciprocal Rank Fusion (RRF):

- **Vector search** — cosine similarity on all-MiniLM-L6-v2 embeddings (384 dimensions)
- **BM25 keyword search** — term-frequency scoring for exact matches

### Git History Search

Semantic search over commit messages, diffs, and metadata:

```rust
use brainwires_knowledge::rag::{RagClient, SearchGitHistoryRequest};

let client = RagClient::new().await?;
let results = client.search_git_history(SearchGitHistoryRequest {
    path: "/my/project".into(),
    query: "authentication refactor".into(),
    limit: Some(10),
    min_score: Some(0.6),
    ..Default::default()
}).await?;
```

### Code Navigation

With the `code-analysis` feature enabled, RagClient provides IDE-like navigation:

- **`find_definition`** — locate where a symbol is defined
- **`find_references`** — find all usages of a symbol
- **`get_call_graph`** — build caller/callee graph for a function

### Configuration

```rust
use brainwires_knowledge::rag::Config;

let config = Config {
    vector_db: VectorDbConfig {
        backend: "lancedb".into(),
        lancedb_path: "~/.brainwires/rag".into(),
        ..Default::default()
    },
    embedding: EmbeddingConfig {
        model_name: "all-MiniLM-L6-v2".into(),
        batch_size: 256,
        ..Default::default()
    },
    indexing: IndexingConfig { .. },
    search: SearchConfig { .. },
    cache: CacheConfig { .. },
};

let client = RagClient::with_config(config).await?;
```

Configuration loads from multiple sources with priority: CLI args > environment variables > config file > defaults.

### MCP Server

A standalone MCP server binary for the RAG subsystem is available at `extras/brainwires-rag-server/`.

## Spectral Subsystem

*Feature: `spectral`*

MSS-inspired spectral subset selection for diverse RAG retrieval. Standard top-k retrieval by cosine similarity produces redundant results. The `SpectralReranker` uses greedy log-determinant maximization to select items that are both relevant AND collectively diverse.

**Algorithm:** Build a relevance-weighted kernel matrix `L_ij = (r_i^lambda) * (r_j^lambda) * cos_sim(v_i, v_j)` and greedily select the subset that maximizes `log det(L_S)`. Achieves a (1-1/e) approximation ratio with O(n*k^2) complexity via incremental Cholesky updates.

```rust
use brainwires_knowledge::spectral::{SpectralReranker, SpectralSelectConfig};

let reranker = SpectralReranker::new(SpectralSelectConfig {
    lambda: 0.5,          // relevance/diversity trade-off (0=diverse, 1=relevant)
    min_candidates: 10,   // skip spectral below this threshold
    ..Default::default()
});

let selected_indices = reranker.rerank(&search_results, &embeddings, 10);
```

The `DiversityReranker` trait allows custom reranking implementations.

## Code Analysis

*Feature: `code-analysis`*

Tree-sitter based AST analysis for symbol extraction and code navigation. Requires `tree-sitter-languages` for parser support.

**Key types:**

| Type | Description |
|------|-------------|
| `Definition` | Symbol definition with location, kind, and visibility |
| `Reference` | Symbol reference with kind (`Call`, `Import`, `TypeRef`, etc.) |
| `CallGraphNode` | Node in a call graph with caller/callee edges |
| `SymbolKind` | `Function`, `Method`, `Class`, `Interface`, `Struct`, `Enum`, etc. |
| `Visibility` | `Public`, `Private`, `Protected`, `Internal` |

**Modules:**

- `repomap` — AST-based symbol extraction across a repository
- `storage` — LanceDB persistence for code relations

## Integration

Use via the `brainwires` facade crate:

```toml
[dependencies]
brainwires = { version = "0.11", features = ["cognition"] }
```

Or depend on `brainwires-knowledge` directly:

```toml
[dependencies]
brainwires-knowledge = { version = "0.11", features = ["native"] }
```

**Import path migration:**

| Old path | New path |
|----------|----------|
| `brainwires_brain::BrainClient` | `brainwires_knowledge::knowledge::BrainClient` |
| `brainwires_brain::EntityStore` | `brainwires_knowledge::knowledge::EntityStore` |
| `brainwires_prompting::TechniqueLibrary` | `brainwires_knowledge::prompting::TechniqueLibrary` |
| `brainwires_prompting::PromptGenerator` | `brainwires_knowledge::prompting::PromptGenerator` |
| `brainwires_rag::RagClient` | `brainwires_knowledge::rag::RagClient` |
| `brainwires_rag::Config` | `brainwires_knowledge::rag::Config` |

Most types are also re-exported at the crate root and in the `prelude` module:

```rust
use brainwires_knowledge::prelude::*;
// BrainClient, EntityStore, PromptGenerator, RagClient, SpectralReranker, etc.
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
