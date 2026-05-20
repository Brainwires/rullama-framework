#![deny(missing_docs)]
//! `brainwires-prompting` — adaptive prompting techniques for the
//! Brainwires Agent Framework.
//!
//! Originally lived in `brainwires-knowledge::prompting`; extracted in
//! Phase 6 of the layout refactor. Pulls only `linfa` / `linfa-clustering`
//! / `ndarray` / `bincode` for the K-means task-clustering pipeline.
//! Optional integration with the BKS/PKS knowledge layer for the
//! adaptive `PromptGenerator` is gated behind the `knowledge` feature.

/// K-means task clustering by semantic-vector similarity.
pub mod clustering;
/// Dynamic prompt generation. With the `knowledge` feature enabled,
/// integrates BKS (Behavioral Knowledge System) / PKS (Personal Knowledge
/// System) / SEAL feedback to adapt outputs over time.
pub mod generator;
/// Technique-effectiveness tracking and BKS promotion logic.
pub mod learning;
/// Library of 15 prompting techniques from the adaptive-selection paper.
pub mod library;
/// SEAL (Self-Evolving Agentic Learning) feedback hook used by `generator`.
pub mod seal;
/// SQLite-backed cluster storage (gated by the `storage` feature).
#[cfg(feature = "storage")]
pub mod storage;
/// Technique enum + per-technique metadata (category / complexity / characteristics).
pub mod techniques;
/// Adaptive temperature optimisation per task cluster.
pub mod temperature;

// ── Public re-exports — preserve the surface previously exposed at
// `brainwires_knowledge::{TaskCluster, PromptingTechnique, …}`.

pub use clustering::{TaskCluster, TaskClusterManager, cosine_similarity};
pub use library::TechniqueLibrary;
pub use seal::SealProcessingResult;
pub use techniques::{
    ComplexityLevel, PromptingTechnique, TaskCharacteristic, TechniqueCategory, TechniqueMetadata,
};

pub use generator::{GeneratedPrompt, PromptGenerator};
pub use learning::{ClusterSummary, PromptingLearningCoordinator, TechniqueStats};
pub use temperature::{TemperatureOptimizer, TemperaturePerformance};

#[cfg(feature = "storage")]
pub use storage::{ClusterStorage, StorageStats};
