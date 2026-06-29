#![deny(missing_docs)]
//! # Brainwires Datasets
//!
//! Training data pipelines for the Brainwires Agent Framework.
//!
//! Provides JSONL I/O, tokenization, deduplication, format conversion, and
//! dataset management for cloud and local model fine-tuning workflows.

/// Dataset trait and concrete dataset implementations.
pub mod dataset;
/// Error types for dataset operations.
pub mod error;
/// Format converters for various fine-tuning providers.
pub mod format;
/// JSONL reader and writer for streaming I/O.
pub mod jsonl;
/// Data quality validation, statistics, and deduplication.
pub mod quality;
/// Train/eval splitting, curriculum ordering, and sampling utilities.
pub mod sampling;
/// Tokenizer abstractions and implementations.
pub mod tokenizer;
/// Core training data types (messages, examples, preference pairs).
pub mod types;

// Re-export core types
pub use dataset::{Dataset, InstructDataset, PreferenceDataset};
pub use error::{DatasetError, DatasetResult};
pub use format::{
    AlpacaFormat, ChatMlFormat, FormatConverter, OpenAiFormat, PreferenceConverter, ShareGptFormat,
    TogetherFormat, detect_format,
};
pub use jsonl::{
    JsonlReader, JsonlWriter, read_jsonl, read_jsonl_preferences, write_jsonl,
    write_jsonl_preferences,
};
pub use quality::{
    DataValidator, DatasetStats, HistogramBucket, IssueSeverity, PreferenceStats, RoleCounts,
    ValidationIssue, ValidationReport, ValidatorConfig, compute_preference_stats, compute_stats,
};
pub use sampling::{
    PreferenceSplitResult, SplitConfig, SplitResult, curriculum_order, preference_curriculum_order,
    preference_sample_n, preference_train_eval_split, sample_n, train_eval_split,
};
pub use types::{DataFormat, PreferencePair, TrainingExample, TrainingMessage, TrainingRole};

// Feature-gated re-exports
#[cfg(feature = "datasets-hf-tokenizer")]
pub use tokenizer::HfTokenizer;

#[cfg(feature = "datasets-tiktoken")]
pub use tokenizer::TiktokenTokenizer;

#[cfg(feature = "datasets-dedup")]
pub use quality::{Deduplicator, exact_dedup, exact_dedup_preferences};

pub use tokenizer::Tokenizer;
