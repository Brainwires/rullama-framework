//! SEAL (Self-Explanatory Adaptive Learning) bridge types
//!
//! These are minimal type definitions for SEAL integration points.
//! The full SEAL implementation lives in brainwires-cli (backend-specific).

use serde::{Deserialize, Serialize};

/// Result of SEAL query processing
///
/// Contains quality metrics from SEAL's self-explanatory analysis.
/// Used by clustering and prompt generation for quality-aware technique selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealProcessingResult {
    /// Quality score of the resolved query (0.0-1.0)
    pub quality_score: f32,
    /// The resolved/refined query from SEAL processing
    pub resolved_query: String,
}

impl SealProcessingResult {
    /// Create a new SEAL processing result
    pub fn new(quality_score: f32, resolved_query: impl Into<String>) -> Self {
        Self {
            quality_score,
            resolved_query: resolved_query.into(),
        }
    }
}
