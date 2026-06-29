//! Personal Fact data structures
//!
//! Defines the core types for representing personal facts about the user -
//! preferences, context, capabilities, and other user-specific knowledge.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A personal fact about the user
///
/// Examples:
/// - "User prefers Rust over Python"
/// - "Current project is rullama-cli"
/// - "User is a Senior Engineer at Anthropic"
/// - "User uses VSCode with Vim keybindings"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalFact {
    /// Unique identifier (UUID)
    pub id: String,

    /// Category of fact (Identity, Preference, etc.)
    pub category: PersonalFactCategory,

    /// Key for the fact (e.g., "preferred_language", "current_project")
    pub key: String,

    /// The actual fact content
    pub value: String,

    /// Optional context when this applies
    pub context: Option<String>,

    // Confidence tracking (EMA-weighted)
    /// Current confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Number of times this fact has been reinforced
    pub reinforcements: u32,

    /// Number of times this fact has been contradicted
    pub contradictions: u32,

    /// Last time this fact was used (Unix timestamp)
    pub last_used: i64,

    // Provenance
    /// When this fact was created (Unix timestamp)
    pub created_at: i64,

    /// When this fact was last updated (Unix timestamp)
    pub updated_at: i64,

    /// How this fact was learned
    pub source: PersonalFactSource,

    /// Server-side version for sync conflict resolution
    #[serde(default)]
    pub version: u64,

    /// Whether this fact has been deleted (soft delete)
    #[serde(default)]
    pub deleted: bool,

    /// Whether this fact should never sync to server (privacy)
    #[serde(default)]
    pub local_only: bool,
}

impl PersonalFact {
    /// Create a new personal fact
    pub fn new(
        category: PersonalFactCategory,
        key: String,
        value: String,
        context: Option<String>,
        source: PersonalFactSource,
        local_only: bool,
    ) -> Self {
        let now = Utc::now().timestamp();
        let initial_confidence = source.initial_confidence();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            category,
            key,
            value,
            context,
            confidence: initial_confidence,
            reinforcements: 0,
            contradictions: 0,
            last_used: now,
            created_at: now,
            updated_at: now,
            source,
            version: 1,
            deleted: false,
            local_only,
        }
    }

    /// Reinforce this fact (successful use)
    ///
    /// Updates confidence using EMA: new = alpha * 1.0 + (1-alpha) * old
    pub fn reinforce(&mut self, ema_alpha: f32) {
        self.reinforcements += 1;
        self.last_used = Utc::now().timestamp();
        self.updated_at = self.last_used;
        self.confidence = ema_alpha * 1.0 + (1.0 - ema_alpha) * self.confidence;
        self.confidence = self.confidence.min(1.0);
        self.version += 1;
    }

    /// Contradict this fact (user provided different info)
    ///
    /// Updates confidence using EMA: new = alpha * 0.0 + (1-alpha) * old
    pub fn contradict(&mut self, ema_alpha: f32) {
        self.contradictions += 1;
        self.last_used = Utc::now().timestamp();
        self.updated_at = self.last_used;
        self.confidence = ema_alpha * 0.0 + (1.0 - ema_alpha) * self.confidence;
        self.version += 1;
    }

    /// Update the fact's value
    pub fn update_value(&mut self, new_value: String) {
        self.value = new_value;
        self.last_used = Utc::now().timestamp();
        self.updated_at = self.last_used;
        self.version += 1;
    }

    /// Apply time-based decay to confidence based on category
    pub fn apply_decay(&mut self) {
        let decay_days = self.category.decay_days();
        let now = Utc::now().timestamp();
        let days_since_use = (now - self.last_used) as f64 / 86400.0;

        // Only decay after category-specific threshold
        if days_since_use > decay_days as f64 {
            let excess_days = days_since_use - decay_days as f64;
            let decay_factor = 0.99_f64.powf(excess_days);
            self.confidence = (self.confidence as f64 * decay_factor) as f32;
        }
    }

    /// Get the decayed confidence (without modifying)
    pub fn decayed_confidence(&self) -> f32 {
        let decay_days = self.category.decay_days();
        let now = Utc::now().timestamp();
        let days_since_use = (now - self.last_used) as f64 / 86400.0;

        if days_since_use > decay_days as f64 {
            let excess_days = days_since_use - decay_days as f64;
            let decay_factor = 0.99_f64.powf(excess_days);
            (self.confidence as f64 * decay_factor) as f32
        } else {
            self.confidence
        }
    }

    /// Check if this fact is still reliable (above threshold)
    pub fn is_reliable(&self, min_confidence: f32) -> bool {
        self.decayed_confidence() >= min_confidence
    }

    /// Mark as deleted (soft delete)
    pub fn delete(&mut self) {
        self.deleted = true;
        self.updated_at = Utc::now().timestamp();
        self.version += 1;
    }

    /// Get a display string for context injection
    pub fn to_context_string(&self) -> String {
        if let Some(ref ctx) = self.context {
            format!("{}: {} (when {})", self.key, self.value, ctx)
        } else {
            format!("{}: {}", self.key, self.value)
        }
    }
}

impl fmt::Display for PersonalFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.0}%] {}/{}: {}",
            self.confidence * 100.0,
            self.category,
            self.key,
            self.value
        )
    }
}

/// Category of personal fact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonalFactCategory {
    /// Name, role, team, organization (decay: 180 days)
    Identity,

    /// Coding style, communication tone, tool preferences (decay: 60 days)
    Preference,

    /// Skills, languages, frameworks known (decay: 90 days)
    Capability,

    /// Current project, recent work, active files (decay: 14 days)
    Context,

    /// Limitations, access restrictions, time zones (decay: 90 days)
    Constraint,

    /// Connections between facts, Zettelkasten-style (decay: 60 days)
    Relationship,

    /// User's preferred ambiguity types for disambiguation (decay: 60 days)
    /// Example: "User prefers SEMANTIC type for technical term disambiguation"
    AmbiguityTypePreference,
}

impl PersonalFactCategory {
    /// Get all categories
    pub fn all() -> &'static [PersonalFactCategory] {
        &[
            PersonalFactCategory::Identity,
            PersonalFactCategory::Preference,
            PersonalFactCategory::Capability,
            PersonalFactCategory::Context,
            PersonalFactCategory::Constraint,
            PersonalFactCategory::Relationship,
            PersonalFactCategory::AmbiguityTypePreference,
        ]
    }

    /// Get the decay days for this category
    pub fn decay_days(&self) -> u32 {
        match self {
            PersonalFactCategory::Identity => 180,
            PersonalFactCategory::Preference => 60,
            PersonalFactCategory::Capability => 90,
            PersonalFactCategory::Context => 14,
            PersonalFactCategory::Constraint => 90,
            PersonalFactCategory::Relationship => 60,
            PersonalFactCategory::AmbiguityTypePreference => 60,
        }
    }

    /// Get a short code for the category
    pub fn code(&self) -> &'static str {
        match self {
            PersonalFactCategory::Identity => "id",
            PersonalFactCategory::Preference => "pref",
            PersonalFactCategory::Capability => "cap",
            PersonalFactCategory::Context => "ctx",
            PersonalFactCategory::Constraint => "limit",
            PersonalFactCategory::Relationship => "rel",
            PersonalFactCategory::AmbiguityTypePreference => "amb",
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            PersonalFactCategory::Identity => "Identity (name, role, organization)",
            PersonalFactCategory::Preference => "Preferences (coding style, tools)",
            PersonalFactCategory::Capability => "Capabilities (skills, languages)",
            PersonalFactCategory::Context => "Context (current project, recent work)",
            PersonalFactCategory::Constraint => "Constraints (limitations, restrictions)",
            PersonalFactCategory::Relationship => "Relationships (fact connections)",
            PersonalFactCategory::AmbiguityTypePreference => "Ambiguity Type Preferences (AT-CoT)",
        }
    }
}

impl fmt::Display for PersonalFactCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

impl std::str::FromStr for PersonalFactCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "id" | "identity" => Ok(PersonalFactCategory::Identity),
            "pref" | "preference" => Ok(PersonalFactCategory::Preference),
            "cap" | "capability" => Ok(PersonalFactCategory::Capability),
            "ctx" | "context" => Ok(PersonalFactCategory::Context),
            "limit" | "constraint" => Ok(PersonalFactCategory::Constraint),
            "rel" | "relationship" => Ok(PersonalFactCategory::Relationship),
            _ => Err(format!("Unknown category: {}", s)),
        }
    }
}

/// How a personal fact was learned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonalFactSource {
    /// User directly stated via /profile command (confidence: 0.9)
    ExplicitStatement,

    /// Detected from conversation patterns like "I prefer..." (confidence: 0.7)
    InferredFromBehavior,

    /// From initial profile setup (confidence: 0.85)
    ProfileSetup,

    /// Observed from tool usage patterns (confidence: 0.6)
    SystemObserved,
}

impl PersonalFactSource {
    /// Get the initial confidence for this source type
    pub fn initial_confidence(&self) -> f32 {
        match self {
            PersonalFactSource::ExplicitStatement => 0.9,
            PersonalFactSource::InferredFromBehavior => 0.7,
            PersonalFactSource::ProfileSetup => 0.85,
            PersonalFactSource::SystemObserved => 0.6,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            PersonalFactSource::ExplicitStatement => "Explicitly stated via /profile",
            PersonalFactSource::InferredFromBehavior => "Inferred from conversation",
            PersonalFactSource::ProfileSetup => "From profile setup",
            PersonalFactSource::SystemObserved => "Observed from usage patterns",
        }
    }
}

impl fmt::Display for PersonalFactSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PersonalFactSource::ExplicitStatement => "explicit",
            PersonalFactSource::InferredFromBehavior => "inferred",
            PersonalFactSource::ProfileSetup => "setup",
            PersonalFactSource::SystemObserved => "observed",
        };
        write!(f, "{}", s)
    }
}

/// A pending fact submission (for offline queue)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingFactSubmission {
    /// The fact to submit
    pub fact: PersonalFact,

    /// When this submission was queued
    pub queued_at: i64,

    /// Number of submission attempts
    pub attempts: u32,

    /// Last error message (if any)
    pub last_error: Option<String>,
}

impl PendingFactSubmission {
    /// Create a new pending submission for the given fact.
    pub fn new(fact: PersonalFact) -> Self {
        Self {
            fact,
            queued_at: Utc::now().timestamp(),
            attempts: 0,
            last_error: None,
        }
    }

    /// Record a submission attempt with an optional error.
    pub fn record_attempt(&mut self, error: Option<String>) {
        self.attempts += 1;
        self.last_error = error;
    }
}

/// A reinforcement or contradiction report to send to server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalFactFeedback {
    /// ID of the fact being reported on
    pub fact_id: String,

    /// Whether this is a reinforcement (true) or contradiction (false)
    pub is_reinforcement: bool,

    /// Optional context about the feedback
    pub context: Option<String>,

    /// When this feedback was generated
    pub timestamp: i64,
}

impl PersonalFactFeedback {
    /// Create a reinforcement feedback for the given fact.
    pub fn reinforcement(fact_id: String, context: Option<String>) -> Self {
        Self {
            fact_id,
            is_reinforcement: true,
            context,
            timestamp: Utc::now().timestamp(),
        }
    }

    /// Create a contradiction feedback for the given fact.
    pub fn contradiction(fact_id: String, context: Option<String>) -> Self {
        Self {
            fact_id,
            is_reinforcement: false,
            context,
            timestamp: Utc::now().timestamp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_creation() {
        let fact = PersonalFact::new(
            PersonalFactCategory::Preference,
            "preferred_language".to_string(),
            "Rust".to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            false,
        );

        assert_eq!(fact.category, PersonalFactCategory::Preference);
        assert_eq!(fact.confidence, 0.9); // ExplicitStatement initial
        assert_eq!(fact.reinforcements, 0);
        assert!(!fact.deleted);
        assert!(!fact.local_only);
    }

    #[test]
    fn test_reinforcement() {
        let mut fact = PersonalFact::new(
            PersonalFactCategory::Context,
            "current_project".to_string(),
            "rullama".to_string(),
            None,
            PersonalFactSource::SystemObserved, // 0.6 initial
            false,
        );

        assert_eq!(fact.confidence, 0.6);

        // Reinforce with alpha = 0.1
        fact.reinforce(0.1);

        // new = 0.1 * 1.0 + 0.9 * 0.6 = 0.1 + 0.54 = 0.64
        assert!((fact.confidence - 0.64).abs() < 0.001);
        assert_eq!(fact.reinforcements, 1);
    }

    #[test]
    fn test_contradiction() {
        let mut fact = PersonalFact::new(
            PersonalFactCategory::Preference,
            "editor".to_string(),
            "VSCode".to_string(),
            None,
            PersonalFactSource::ExplicitStatement, // 0.9 initial
            false,
        );

        // Contradict with alpha = 0.1
        fact.contradict(0.1);

        // new = 0.1 * 0.0 + 0.9 * 0.9 = 0 + 0.81 = 0.81
        assert!((fact.confidence - 0.81).abs() < 0.001);
        assert_eq!(fact.contradictions, 1);
    }

    #[test]
    fn test_category_decay_days() {
        assert_eq!(PersonalFactCategory::Identity.decay_days(), 180);
        assert_eq!(PersonalFactCategory::Preference.decay_days(), 60);
        assert_eq!(PersonalFactCategory::Context.decay_days(), 14);
        assert_eq!(PersonalFactCategory::Capability.decay_days(), 90);
    }

    #[test]
    fn test_category_parsing() {
        assert_eq!(
            "id".parse::<PersonalFactCategory>().unwrap(),
            PersonalFactCategory::Identity
        );
        assert_eq!(
            "pref".parse::<PersonalFactCategory>().unwrap(),
            PersonalFactCategory::Preference
        );
        assert_eq!(
            "ctx".parse::<PersonalFactCategory>().unwrap(),
            PersonalFactCategory::Context
        );
        assert!("invalid".parse::<PersonalFactCategory>().is_err());
    }

    #[test]
    fn test_source_initial_confidence() {
        assert_eq!(
            PersonalFactSource::ExplicitStatement.initial_confidence(),
            0.9
        );
        assert_eq!(
            PersonalFactSource::InferredFromBehavior.initial_confidence(),
            0.7
        );
        assert_eq!(PersonalFactSource::ProfileSetup.initial_confidence(), 0.85);
        assert_eq!(PersonalFactSource::SystemObserved.initial_confidence(), 0.6);
    }

    #[test]
    fn test_fact_display() {
        let fact = PersonalFact::new(
            PersonalFactCategory::Preference,
            "language".to_string(),
            "Rust".to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            false,
        );

        let display = format!("{}", fact);
        assert!(display.contains("90%"));
        assert!(display.contains("pref"));
        assert!(display.contains("language"));
        assert!(display.contains("Rust"));
    }

    #[test]
    fn test_context_string() {
        let fact = PersonalFact::new(
            PersonalFactCategory::Preference,
            "framework".to_string(),
            "React".to_string(),
            Some("frontend projects".to_string()),
            PersonalFactSource::ExplicitStatement,
            false,
        );

        assert_eq!(
            fact.to_context_string(),
            "framework: React (when frontend projects)"
        );

        let fact_no_context = PersonalFact::new(
            PersonalFactCategory::Identity,
            "name".to_string(),
            "John".to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            false,
        );

        assert_eq!(fact_no_context.to_context_string(), "name: John");
    }
}
