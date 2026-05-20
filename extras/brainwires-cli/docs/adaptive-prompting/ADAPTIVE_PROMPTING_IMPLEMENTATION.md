# Adaptive Prompting

Brainwires CLI implements **Adaptive Prompting** — automatic selection of the most effective prompting technique for each task based on task characteristics, SEAL quality signals, and learned user/collective preferences.

This feature is based on the paper *"Adaptive Selection of Prompting Techniques"* (arXiv:2510.18162).

---

## Overview

Rather than applying the same system prompt template to every task, the adaptive prompting system:

1. **Classifies** the task by embedding its description and matching it to a known task cluster
2. **Selects** the best combination of prompting techniques for that cluster
3. **Personalizes** technique selection using PKS (user preferences) and BKS (collective learning)
4. **Filters** technique complexity based on SEAL's query quality score
5. **Generates** a dynamic system prompt combining the selected techniques
6. **Learns** which techniques perform well and promotes them into shared knowledge

---

## Module Layout

> **Note:** The prompting module lives in the `brainwires-knowledge` framework crate (a
> dependency of the CLI), not in the CLI's own `src/` directory.

```
brainwires-knowledge crate / src/prompting/
├── techniques.rs    # 15 prompting technique definitions and metadata
├── library.rs       # TechniqueLibrary — BKS-backed technique catalog
├── clustering.rs    # TaskClusterManager — k-means cluster matching
├── generator.rs     # PromptGenerator — technique selection and prompt composition
├── learning.rs      # Effectiveness tracking; BKS/PKS promotion pipeline
├── temperature.rs   # Adaptive temperature per cluster
└── storage.rs       # SQLite persistence for clusters and performance data
```

---

## Prompting Techniques

All 15 techniques from the paper are implemented:

| Category | Techniques |
|----------|------------|
| Role Assignment | Role Playing |
| Emotional Stimulus | Emotion Prompting, Stress Prompting |
| Reasoning | Chain-of-Thought, Logic-of-Thought, Least-to-Most, Thread-of-Thought, Plan-and-Solve, Skeleton-of-Thought, Scratchpad Prompting |
| Others | Decomposed Prompting, Ignore Irrelevant Conditions, Highlighted CoT, Skills-in-Context, Automatic Information Filtering |

### SEAL Quality Mapping

Technique complexity is gated on SEAL's query quality score to avoid over-engineering simple tasks:

```
Quality < 0.5  → Simple techniques only (CoT, Role Playing, Emotion)
Quality 0.5–0.8 → Moderate techniques (Plan-and-Solve, Least-to-Most)
Quality > 0.8  → Advanced techniques (Logic-of-Thought, Skills-in-Context)
```

---

## Task Clustering

The `TaskClusterManager` uses k-means to group similar tasks:

- Optimal cluster count is determined by silhouette score maximization
- Task embeddings are computed with the existing `EmbeddingProvider` (384-dim FastEmbed, LRU-cached)
- SEAL query cores are stored per cluster; high-quality SEAL results get a 10% similarity boost
- Each cluster tracks its recommended complexity level based on average SEAL quality

---

## Technique Selection Algorithm

The `PromptGenerator` applies a priority-based selection:

```
PKS (user preference) > BKS (collective learning) > Cluster default
```

Per-task selection:
1. **Role Playing** — always included (paper's baseline rule)
2. **Emotional Stimulus** — select 1 using priority system above
3. **Reasoning** — select 1 based on SEAL quality complexity level
4. **Others** — include 0–1 if SEAL quality > 0.6

Target: 3–4 techniques per prompt (paper-compliant).

---

## Integration Architecture

```
User Query
    │
    ▼
SEAL Processor
    ├─ Coreference resolution
    ├─ Query core extraction
    └─ Quality scoring (0.0–1.0)
    │
    ▼
SealKnowledgeCoordinator
    ├─ BKS context retrieval
    ├─ PKS context retrieval
    └─ Confidence harmonization
    │
    ▼
OrchestratorAgent.build_system_prompt()
    ├─ IF adaptive_prompts_enabled:
    │   ├─ Embed task → match cluster
    │   ├─ Select techniques (PKS > BKS > cluster default)
    │   ├─ Filter by SEAL quality complexity
    │   └─ Generate dynamic system prompt
    └─ ELSE: static fallback prompt
    │
    ▼
Provider.chat()
    │
    ▼
Learning Pipeline
    ├─ Track technique effectiveness (EMA, α=0.3)
    ├─ Promote to BKS when reliability > 80% with 5+ uses
    └─ Store PKS preferences
```

### Knowledge System Integration Points

| System | Role |
|--------|------|
| SEAL | Query core for classification; quality score for complexity filtering; 10% similarity boost for high quality |
| BKS | Shared technique effectiveness via `TruthCategory::PromptingTechnique`; temperature preferences |
| PKS | Per-user technique preferences (highest priority override) |

---

## Temperature Optimization

`temperature.rs` adapts the generation temperature per task cluster:

- Selection: BKS shared value > local learned > heuristic default
- Paper-compliant defaults: `0.0` for logical/mathematical tasks, `1.3` for linguistic/creative tasks
- Effective temperatures are promoted to BKS after consistent good performance

---

## Persistence

`storage.rs` provides SQLite-backed persistence for:
- Task cluster centroids and SEAL metrics
- Technique effectiveness statistics per cluster
- Temperature performance data

Database location: `~/.brainwires/adaptive_prompting.db`

---

## OrchestratorAgent API

```rust
// Enable adaptive prompting
pub fn enable_adaptive_prompting(
    &mut self,
    generator: PromptGenerator,
    embedding_provider: Arc<CachedEmbeddingProvider>,
)

// Disable (falls back to static prompt)
pub fn disable_adaptive_prompting(&mut self)

pub fn is_adaptive_prompting_enabled(&self) -> bool

// Access last-generated prompt for learning/debugging
pub fn last_generated_prompt(&self) -> Option<&GeneratedPrompt>
```

---

## Usage Example

```rust
use brainwires_cli::prompting::{TechniqueLibrary, TaskClusterManager, PromptGenerator};
use brainwires_cli::prompting::storage::ClusterStorage;
use brainwires_cli::storage::embeddings::CachedEmbeddingProvider;
use std::sync::Arc;

// Initialize components
let embedding_provider = Arc::new(CachedEmbeddingProvider::new()?);

let library = TechniqueLibrary::new()
    .with_bks(bks_cache.clone());

let mut cluster_manager = TaskClusterManager::new();
let storage = ClusterStorage::new("~/.brainwires/adaptive_prompting.db")?;
for cluster in storage.load_clusters()? {
    cluster_manager.add_cluster(cluster);
}

let generator = PromptGenerator::new(library, cluster_manager)
    .with_knowledge(bks_cache, pks_cache);

// Enable on orchestrator
orchestrator.enable_adaptive_prompting(generator, embedding_provider);

// After execution, inspect what was selected
if let Some(prompt_info) = orchestrator.last_generated_prompt() {
    println!("Cluster: {}", prompt_info.cluster_id);
    println!("Techniques: {:?}", prompt_info.techniques);
    println!("SEAL quality: {:.2}", prompt_info.seal_quality);
}
```

If no clusters are loaded, the system falls back to static prompts gracefully.

---

## Performance

**Expected overhead per request:**
- Task embedding + cluster match: < 50 ms
- Technique selection: < 10 ms
- Prompt generation: < 100 ms
- **Total:** < 200 ms

**Paper results (BIG-Bench Extra Hard):**
- Baseline: 24.7% arithmetic mean
- Adaptive prompting: 28.0% (+13.4%)
- Best gains: Object Counting (+59%), Spatial Reasoning (+20%)

---

## Test Coverage

34 unit tests across the prompting module:

| Module | Tests |
|--------|-------|
| `techniques.rs` | 6 (enum variants, metadata, serialization) |
| `library.rs` | 4 (technique library, BKS integration) |
| `clustering.rs` | 4 (k-means, cosine similarity, cluster matching) |
| `generator.rs` | 3 (prompt generation, role inference, task type) |
| `learning.rs` | 6 (effectiveness tracking, BKS promotion) |
| `temperature.rs` | 6 (performance tracking, heuristics, optimization) |
| `storage.rs` | 5 (SQLite CRUD, persistence) |

Integration tests covering the full SEAL → Adaptive → Knowledge flow are tracked in `docs/FUTURE_WORK.md`.
