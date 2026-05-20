# brainwires-prompting

[![Crates.io](https://img.shields.io/crates/v/brainwires-prompting.svg)](https://crates.io/crates/brainwires-prompting)
[![Documentation](https://docs.rs/brainwires-prompting/badge.svg)](https://docs.rs/brainwires-prompting)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Adaptive prompting techniques, K-means task clustering, and temperature
optimization for Brainwires agents.

## What lives here

- `library::TechniqueLibrary` + `techniques::*` — 15-technique library
  from the adaptive-selection paper (chain-of-thought, self-consistency,
  tree-of-thought, …).
- `clustering::TaskCluster` + `TaskClusterManager` — K-means task
  clustering by semantic-vector similarity (linfa-backed).
- `generator::PromptGenerator` — dynamic prompt generation. With the
  `knowledge` feature (default-on) integrates BKS (Behavioral Knowledge
  System) / PKS (Personal Knowledge System) / SEAL feedback to adapt
  outputs over time.
- `learning::PromptingLearningCoordinator` — technique-effectiveness
  tracking + BKS promotion logic.
- `temperature::TemperatureOptimizer` — adaptive temperature optimisation
  per task cluster.
- `seal::SealProcessingResult` — feedback hook used by `generator`.
- `storage::ClusterStorage` (gated by `storage` feature) — SQLite-backed
  cluster persistence.

## Features

| Feature | Default | Notes |
|---|---|---|
| `knowledge` | yes | Pulls `brainwires-knowledge` for BKS/PKS-aware prompt generation |
| `storage` | no | SQLite cluster store (rusqlite) |

Most modules (`generator`, `learning`, `temperature`) reference BKS/PKS
unconditionally, hence the `knowledge` default. The standalone bits
(`clustering`, `library`, `techniques`, `seal`) work without it.

## Usage

```toml
[dependencies]
brainwires-prompting = "0.11"
```

```rust,ignore
use brainwires_prompting::{TechniqueLibrary, PromptingTechnique};

let library = TechniqueLibrary::default();
let technique = library.lookup(PromptingTechnique::ChainOfThought);
```

## See also

- [`brainwires-knowledge`](https://crates.io/crates/brainwires-knowledge) —
  BKS / PKS / brain client (used by `generator`).
- [`brainwires-rag`](https://crates.io/crates/brainwires-rag) — codebase
  indexing + hybrid retrieval (sibling).
- [`brainwires`](https://crates.io/crates/brainwires) — umbrella facade
  with `prompting` feature.

## License

MIT OR Apache-2.0
