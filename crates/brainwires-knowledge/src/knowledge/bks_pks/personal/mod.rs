//! Personal Knowledge System (PKS)
//!
//! Stores user-specific facts, preferences, and evolving profile information.
//! Unlike BKS (shared), PKS is strictly user-scoped and synced server-side for
//! consistency across all CLI/TUI instances.
//!
//! ## Key Concepts
//!
//! - **PersonalFact**: A learned fact about the user (preferences, context, capabilities)
//!   (e.g., "User prefers Rust", "Current project is brainwires-cli")
//!
//! - **Server-Side User-Scoped**: Facts are stored on the Brainwires server with RLS
//!   ensuring users can only access their own data.
//!
//! - **Learning Sources**:
//!   1. Explicit: Users teach via `/profile set` command
//!   2. Implicit: System detects patterns like "I prefer...", "I'm working on..."
//!   3. Observed: System observes from tool usage and conversation patterns
//!
//! ## Categories with Decay Rates
//!
//! - **Identity**: name, role, team, organization (decay: 180 days)
//! - **Preference**: coding_style, communication_tone, tool_preferences (decay: 60 days)
//! - **Capability**: skills, languages, frameworks known (decay: 90 days)
//! - **Context**: current_project, recent_work, active_files (decay: 14 days)
//! - **Constraint**: limitations, access restrictions, time zones (decay: 90 days)
//! - **Relationship**: connections between facts, Zettelkasten-style (decay: 60 days)

pub mod api;
pub mod cache;
pub mod collector;
pub mod fact;
pub mod integration;
pub mod matcher;

pub use api::PersonalKnowledgeApiClient;
pub use cache::PersonalKnowledgeCache;
pub use collector::PersonalFactCollector;
pub use fact::{PersonalFact, PersonalFactCategory, PersonalFactFeedback, PersonalFactSource};
pub use integration::{
    DetectedFact, DetectionSource, PksBackgroundProcessor, PksIntegration, PksRestPoller,
};
pub use matcher::PersonalFactMatcher;

/// Configuration for the Personal Knowledge System
#[derive(Debug, Clone)]
pub struct PersonalKnowledgeSettings {
    /// Master toggle for the personal knowledge system
    pub enabled: bool,

    // Learning sources
    /// Enable explicit learning via /profile command
    pub enable_explicit_learning: bool,
    /// Enable implicit learning from conversation patterns
    pub enable_implicit_learning: bool,
    /// Enable observed learning from tool usage
    pub enable_observed_learning: bool,

    // Thresholds
    /// Minimum confidence to include fact in context (default: 0.5)
    pub min_confidence_to_apply: f32,
    /// Confidence threshold for implicit detection (default: 0.6)
    pub implicit_detection_confidence: f32,

    // Decay
    /// EMA decay factor for confidence updates (default: 0.1)
    pub ema_alpha: f32,

    // Sync
    /// How often to sync with server in seconds (default: 300)
    pub sync_interval_secs: u64,
    /// Maximum queued submissions for offline mode (default: 50)
    pub offline_queue_size: usize,

    // Privacy
    /// Default to local-only for new facts (never sync to server)
    pub default_local_only: bool,

    // Display
    /// Show when personal facts are applied to context
    pub show_applied_facts: bool,
}

impl Default for PersonalKnowledgeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_explicit_learning: true,
            enable_implicit_learning: true,
            enable_observed_learning: true,
            min_confidence_to_apply: 0.5,
            implicit_detection_confidence: 0.6,
            ema_alpha: 0.1,
            sync_interval_secs: 300,
            offline_queue_size: 50,
            default_local_only: false,
            show_applied_facts: false,
        }
    }
}

impl PersonalKnowledgeSettings {
    /// Create settings with all learning sources enabled
    pub fn full() -> Self {
        Self::default()
    }

    /// Create settings with only explicit learning enabled
    pub fn explicit_only() -> Self {
        Self {
            enable_implicit_learning: false,
            enable_observed_learning: false,
            ..Self::default()
        }
    }

    /// Create settings with local-only mode (no server sync)
    pub fn local_only() -> Self {
        Self {
            default_local_only: true,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = PersonalKnowledgeSettings::default();
        assert!(settings.enabled);
        assert!(settings.enable_explicit_learning);
        assert!(settings.enable_implicit_learning);
        assert!(settings.enable_observed_learning);
        assert_eq!(settings.min_confidence_to_apply, 0.5);
        assert!(!settings.default_local_only);
    }

    #[test]
    fn test_explicit_only_settings() {
        let settings = PersonalKnowledgeSettings::explicit_only();
        assert!(settings.enabled);
        assert!(settings.enable_explicit_learning);
        assert!(!settings.enable_implicit_learning);
        assert!(!settings.enable_observed_learning);
    }

    #[test]
    fn test_local_only_settings() {
        let settings = PersonalKnowledgeSettings::local_only();
        assert!(settings.enabled);
        assert!(settings.default_local_only);
    }

    #[test]
    fn test_disabled_settings() {
        let settings = PersonalKnowledgeSettings::disabled();
        assert!(!settings.enabled);
    }
}
