#![deny(missing_docs)]
//! # Brainwires Fine-tune
//!
//! Cloud fine-tuning for the Brainwires Agent Framework.
//!
//! Supports cloud fine-tuning (OpenAI, Together, Fireworks, Anyscale, Bedrock,
//! Vertex) plus dataset pipelines. Local adapter training (LoRA, QLoRA, DoRA)
//! and training-from-scratch live in the sibling `rullama` workspace as
//! `rullama-finetune` and `rullama-training`.

/// Training configuration and hyperparameters.
pub mod config;
/// Training error types.
pub mod error;
/// Training job types and status.
pub mod types;

/// Dataset pipelines (absorbed from brainwires-datasets).
pub mod datasets;

/// Cloud fine-tuning providers.
#[cfg(feature = "cloud")]
pub mod cloud;

// Local adapter training (LoRA/QLoRA/DoRA) lives in `rullama-finetune`
// (sibling workspace at /Users/nightness/Source/Brainwires/rullama).

/// Training job management.
pub mod manager;

// Re-export core types (always available)
pub use config::{AdapterMethod, AlignmentMethod, LoraConfig, LrScheduler, TrainingHyperparams};
pub use error::TrainingError;
pub use types::{
    DatasetId, TrainingJobId, TrainingJobStatus, TrainingJobSummary, TrainingMetrics,
    TrainingProgress,
};

#[cfg(feature = "cloud")]
pub use cloud::{CloudFineTuneConfig, FineTuneProvider, FineTuneProviderFactory};

pub use manager::TrainingManager;
