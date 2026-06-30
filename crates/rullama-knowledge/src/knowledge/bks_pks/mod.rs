//! rullama Knowledge - Behavioral and personal knowledge systems
//!
//! This module provides two knowledge systems for the rullama:
//!
//! ## Behavioral Knowledge System (BKS)
//!
//! A collective intelligence system that learns universal behavioral truths from experience
//! across all rullama clients. This is NOT about personal preferences - it's about
//! objectively better ways to accomplish tasks that are shared server-side.
//!
//! - **BehavioralTruth**: Learned rules about how to better accomplish tasks
//! - **BehavioralKnowledgeCache**: SQLite-backed local cache with server sync
//! - **LearningCollector**: Detects learning signals from conversations
//! - **ContextMatcher**: Matches truths to current context
//! - **TruthInferenceEngine**: Infers new truths from patterns
//!
//! ## Personal Knowledge System (PKS)
//!
//! Stores user-specific facts, preferences, and evolving profile information.
//! Unlike BKS (shared), PKS is strictly user-scoped.
//!
//! - **PersonalFact**: Learned facts about the user (preferences, context, capabilities)
//! - **PersonalKnowledgeCache**: SQLite-backed local cache
//! - **PersonalFactCollector**: Detects personal facts from conversations
//! - **PersonalFactMatcher**: Matches facts to current context

// Re-export core types
pub use rullama_core;

// ── Behavioral Knowledge System ────────────────────────────────────────────

pub mod api;
pub mod cache;
pub mod collector;
pub mod inference;
pub mod matcher;
pub mod truth;

// ── Personal Knowledge System ──────────────────────────────────────────────

pub mod personal;

// ── Re-exports ─────────────────────────────────────────────────────────────

// BKS types
pub use api::KnowledgeApiClient;
pub use cache::BehavioralKnowledgeCache;
pub use collector::{LearningCollector, detect_correction};
pub use inference::TruthInferenceEngine;
pub use matcher::ContextMatcher;
pub use truth::{BehavioralTruth, TruthCategory, TruthSource};

// PKS types
pub use personal::{
    PersonalFact, PersonalFactCategory, PersonalFactCollector, PersonalFactMatcher,
    PersonalFactSource, PersonalKnowledgeApiClient, PersonalKnowledgeCache,
    PersonalKnowledgeSettings,
};

/// Configuration for the Behavioral Knowledge System
#[derive(Debug, Clone)]
pub struct KnowledgeSettings {
    /// Master toggle for the knowledge system
    pub enabled: bool,

    // Learning sources
    /// Enable explicit learning via /learn command
    pub enable_explicit_learning: bool,
    /// Enable implicit learning from conversation corrections
    pub enable_implicit_learning: bool,
    /// Enable aggressive learning from success/failure patterns
    pub enable_aggressive_learning: bool,

    // Thresholds
    /// Minimum confidence to inject truth into prompt (default: 0.5)
    pub min_confidence_to_apply: f32,
    /// Minimum confidence to prompt about conflicts (default: 0.7)
    pub min_confidence_to_prompt: f32,
    /// Number of failures before detecting a pattern (default: 3)
    pub failure_threshold: u32,

    // Decay
    /// EMA decay factor for confidence updates (default: 0.1)
    pub ema_alpha: f32,
    /// Days of non-use before decay starts (default: 30)
    pub decay_days: u32,

    // Sync
    /// How often to sync with server in seconds (default: 300)
    pub sync_interval_secs: u64,
    /// Maximum queued submissions for offline mode (default: 100)
    pub offline_queue_size: usize,

    // Display
    /// Show when truths are applied to prompts
    pub show_applied_truths: bool,
    /// Ask user about conflicts between truths and instructions
    pub show_conflict_prompts: bool,
}

impl Default for KnowledgeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_explicit_learning: true,
            enable_implicit_learning: true,
            enable_aggressive_learning: true,
            min_confidence_to_apply: 0.5,
            min_confidence_to_prompt: 0.7,
            failure_threshold: 3,
            ema_alpha: 0.1,
            decay_days: 30,
            sync_interval_secs: 300,
            offline_queue_size: 100,
            show_applied_truths: true,
            show_conflict_prompts: true,
        }
    }
}

impl KnowledgeSettings {
    /// Create settings with all learning sources enabled
    pub fn full() -> Self {
        Self::default()
    }

    /// Create settings with only explicit learning enabled
    pub fn explicit_only() -> Self {
        Self {
            enable_implicit_learning: false,
            enable_aggressive_learning: false,
            ..Self::default()
        }
    }

    /// Create disabled settings
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::KnowledgeSettings;
    pub use super::cache::BehavioralKnowledgeCache;
    pub use super::collector::LearningCollector;
    pub use super::inference::TruthInferenceEngine;
    pub use super::matcher::ContextMatcher;
    pub use super::personal::PersonalKnowledgeCache;
    pub use super::personal::{
        PersonalFact, PersonalFactCategory, PersonalFactCollector, PersonalFactMatcher,
        PersonalFactSource, PersonalKnowledgeSettings,
    };
    pub use super::truth::{BehavioralTruth, TruthCategory, TruthSource};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = KnowledgeSettings::default();
        assert!(settings.enabled);
        assert!(settings.enable_explicit_learning);
        assert!(settings.enable_implicit_learning);
        assert!(settings.enable_aggressive_learning);
        assert_eq!(settings.min_confidence_to_apply, 0.5);
        assert_eq!(settings.failure_threshold, 3);
    }

    #[test]
    fn test_explicit_only_settings() {
        let settings = KnowledgeSettings::explicit_only();
        assert!(settings.enabled);
        assert!(settings.enable_explicit_learning);
        assert!(!settings.enable_implicit_learning);
        assert!(!settings.enable_aggressive_learning);
    }

    #[test]
    fn test_disabled_settings() {
        let settings = KnowledgeSettings::disabled();
        assert!(!settings.enabled);
    }
}
