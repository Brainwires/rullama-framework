/// Dataset statistics computation.
pub mod stats;
/// Dataset validation rules and reporting.
pub mod validator;

/// MinHash-based and exact deduplication.
#[cfg(feature = "datasets-dedup")]
pub mod dedup;

pub use stats::{
    DatasetStats, HistogramBucket, PreferenceStats, RoleCounts, compute_preference_stats,
    compute_stats,
};
pub use validator::{
    DataValidator, IssueSeverity, ValidationIssue, ValidationReport, ValidatorConfig,
};

#[cfg(feature = "datasets-dedup")]
pub use dedup::{Deduplicator, exact_dedup, exact_dedup_preferences};
