//! Behavioral Truth data structures
//!
//! Defines the core types for representing learned behavioral truths -
//! universal knowledge about better ways to accomplish tasks.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A learned behavioral truth - not a preference, but an objective improvement
///
/// Examples:
/// - "pm2 logs requires --nostream flag to avoid blocking"
/// - "For long-running tasks, spawn a dedicated monitoring agent instead of inline polling"
/// - "Use cargo-watch instead of a manual loop for cargo build in watch mode"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralTruth {
    /// Unique identifier (UUID)
    pub id: String,

    /// Category of truth (CommandUsage, TaskStrategy, etc.)
    pub category: TruthCategory,

    /// Context pattern - when this truth applies
    /// e.g., "pm2 logs", "long-running task monitoring", "cargo build watch"
    pub context_pattern: String,

    /// Human-readable rule
    /// e.g., "Use --nostream flag with pm2 logs to avoid blocking"
    pub rule: String,

    /// Rationale explaining why this is better (for conflict resolution)
    /// e.g., "pm2 logs streams indefinitely by default, blocking the terminal"
    pub rationale: String,

    // Confidence tracking (EMA-weighted)
    /// Current confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Number of times this truth has been reinforced (successful use)
    pub reinforcements: u32,

    /// Number of times this truth has been contradicted
    pub contradictions: u32,

    /// Last time this truth was used (Unix timestamp)
    pub last_used: i64,

    // Provenance
    /// When this truth was created (Unix timestamp)
    pub created_at: i64,

    /// Anonymous user/client ID that created this truth
    pub created_by: Option<String>,

    /// How this truth was learned
    pub source: TruthSource,

    /// Server-side version for sync conflict resolution
    #[serde(default)]
    pub version: u64,

    /// Whether this truth has been deleted (soft delete)
    #[serde(default)]
    pub deleted: bool,
}

impl BehavioralTruth {
    /// Create a new behavioral truth
    pub fn new(
        category: TruthCategory,
        context_pattern: String,
        rule: String,
        rationale: String,
        source: TruthSource,
        created_by: Option<String>,
    ) -> Self {
        let now = Utc::now().timestamp();
        let initial_confidence = source.initial_confidence();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            category,
            context_pattern,
            rule,
            rationale,
            confidence: initial_confidence,
            reinforcements: 0,
            contradictions: 0,
            last_used: now,
            created_at: now,
            created_by,
            source,
            version: 1,
            deleted: false,
        }
    }

    /// Reinforce this truth (successful use)
    ///
    /// Updates confidence using EMA: new = α × 1.0 + (1-α) × old
    pub fn reinforce(&mut self, ema_alpha: f32) {
        self.reinforcements += 1;
        self.last_used = Utc::now().timestamp();
        self.confidence = ema_alpha * 1.0 + (1.0 - ema_alpha) * self.confidence;
        self.confidence = self.confidence.min(1.0); // Cap at 1.0
        self.version += 1;
    }

    /// Contradict this truth (user chose differently)
    ///
    /// Updates confidence using EMA: new = α × 0.0 + (1-α) × old
    pub fn contradict(&mut self, ema_alpha: f32) {
        self.contradictions += 1;
        self.last_used = Utc::now().timestamp();
        self.confidence = ema_alpha * 0.0 + (1.0 - ema_alpha) * self.confidence;
        self.version += 1;
    }

    /// Apply time-based decay to confidence
    ///
    /// Formula: confidence × (0.99 ^ days_since_last_use)
    pub fn apply_decay(&mut self, decay_start_days: u32) {
        let now = Utc::now().timestamp();
        let days_since_use = (now - self.last_used) as f64 / 86400.0;

        // Only decay after threshold
        if days_since_use > decay_start_days as f64 {
            let decay_days = days_since_use - decay_start_days as f64;
            let decay_factor = 0.99_f64.powf(decay_days);
            self.confidence = (self.confidence as f64 * decay_factor) as f32;
        }
    }

    /// Get the decayed confidence (without modifying)
    pub fn decayed_confidence(&self, decay_start_days: u32) -> f32 {
        let now = Utc::now().timestamp();
        let days_since_use = (now - self.last_used) as f64 / 86400.0;

        if days_since_use > decay_start_days as f64 {
            let decay_days = days_since_use - decay_start_days as f64;
            let decay_factor = 0.99_f64.powf(decay_days);
            (self.confidence as f64 * decay_factor) as f32
        } else {
            self.confidence
        }
    }

    /// Check if this truth is still reliable (above threshold)
    pub fn is_reliable(&self, min_confidence: f32, decay_start_days: u32) -> bool {
        self.decayed_confidence(decay_start_days) >= min_confidence
    }

    /// Get the success rate (reinforcements / total uses)
    pub fn success_rate(&self) -> f32 {
        let total = self.reinforcements + self.contradictions;
        if total == 0 {
            0.5 // Neutral when no data
        } else {
            self.reinforcements as f32 / total as f32
        }
    }

    /// Mark as deleted (soft delete)
    pub fn delete(&mut self) {
        self.deleted = true;
        self.version += 1;
    }
}

impl fmt::Display for BehavioralTruth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.0}%] {}: {}",
            self.confidence * 100.0,
            self.category,
            self.rule
        )
    }
}

/// Category of behavioral truth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthCategory {
    /// CLI flags, arguments (e.g., "pm2 logs --nostream")
    CommandUsage,

    /// How to approach tasks (spawn monitor vs inline poll)
    TaskStrategy,

    /// Tool-specific knowledge (timeouts, limitations)
    ToolBehavior,

    /// What to do when X fails
    ErrorRecovery,

    /// Context window, parallelism decisions
    ResourceManagement,

    /// Anti-patterns to avoid
    PatternAvoidance,

    /// Adaptive prompting technique effectiveness
    PromptingTechnique,

    /// AT-CoT clarifying question effectiveness
    /// Example: "For cache implementation queries, SEMANTIC+SPECIFY achieves 87% success"
    ClarifyingQuestions,
}

impl TruthCategory {
    /// Get all categories
    pub fn all() -> &'static [TruthCategory] {
        &[
            TruthCategory::CommandUsage,
            TruthCategory::TaskStrategy,
            TruthCategory::ToolBehavior,
            TruthCategory::ErrorRecovery,
            TruthCategory::ResourceManagement,
            TruthCategory::PatternAvoidance,
            TruthCategory::PromptingTechnique,
            TruthCategory::ClarifyingQuestions,
        ]
    }

    /// Get a short code for the category
    pub fn code(&self) -> &'static str {
        match self {
            TruthCategory::CommandUsage => "cmd",
            TruthCategory::TaskStrategy => "task",
            TruthCategory::ToolBehavior => "tool",
            TruthCategory::ErrorRecovery => "error",
            TruthCategory::ResourceManagement => "resource",
            TruthCategory::PatternAvoidance => "avoid",
            TruthCategory::PromptingTechnique => "prompt",
            TruthCategory::ClarifyingQuestions => "clarify",
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            TruthCategory::CommandUsage => "CLI flags and arguments",
            TruthCategory::TaskStrategy => "Task execution strategies",
            TruthCategory::ToolBehavior => "Tool-specific knowledge",
            TruthCategory::ErrorRecovery => "Error recovery patterns",
            TruthCategory::ResourceManagement => "Resource management",
            TruthCategory::PatternAvoidance => "Anti-patterns to avoid",
            TruthCategory::PromptingTechnique => "Prompting technique effectiveness",
            TruthCategory::ClarifyingQuestions => "Clarifying question effectiveness",
        }
    }
}

impl fmt::Display for TruthCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

impl std::str::FromStr for TruthCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cmd" | "command" | "command_usage" | "commandusage" => Ok(TruthCategory::CommandUsage),
            "task" | "strategy" | "task_strategy" | "taskstrategy" => {
                Ok(TruthCategory::TaskStrategy)
            }
            "tool" | "behavior" | "tool_behavior" | "toolbehavior" => {
                Ok(TruthCategory::ToolBehavior)
            }
            "error" | "recovery" | "error_recovery" | "errorrecovery" => {
                Ok(TruthCategory::ErrorRecovery)
            }
            "resource" | "management" | "resource_management" | "resourcemanagement" => {
                Ok(TruthCategory::ResourceManagement)
            }
            "avoid" | "pattern" | "pattern_avoidance" | "patternavoidance" => {
                Ok(TruthCategory::PatternAvoidance)
            }
            _ => Err(format!("Unknown category: {}", s)),
        }
    }
}

/// How a truth was learned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthSource {
    /// User explicitly taught via /learn command
    ExplicitCommand,

    /// User corrected agent mid-conversation
    ConversationCorrection,

    /// Detected from successful outcomes
    SuccessPattern,

    /// Detected from failures (what NOT to do)
    FailurePattern,
}

impl TruthSource {
    /// Get the initial confidence for this source type
    pub fn initial_confidence(&self) -> f32 {
        match self {
            TruthSource::ExplicitCommand => 0.8, // User explicitly taught
            TruthSource::ConversationCorrection => 0.6, // Implicit but clear intent
            TruthSource::SuccessPattern => 0.4,  // Needs reinforcement
            TruthSource::FailurePattern => 0.5,  // Negative knowledge is valuable
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            TruthSource::ExplicitCommand => "Explicitly taught via /learn",
            TruthSource::ConversationCorrection => "Learned from conversation correction",
            TruthSource::SuccessPattern => "Detected from success patterns",
            TruthSource::FailurePattern => "Detected from failure patterns",
        }
    }
}

impl fmt::Display for TruthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TruthSource::ExplicitCommand => "explicit",
            TruthSource::ConversationCorrection => "correction",
            TruthSource::SuccessPattern => "success",
            TruthSource::FailurePattern => "failure",
        };
        write!(f, "{}", s)
    }
}

/// A pending truth submission (for offline queue)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTruthSubmission {
    /// The truth to submit
    pub truth: BehavioralTruth,

    /// When this submission was queued
    pub queued_at: i64,

    /// Number of submission attempts
    pub attempts: u32,

    /// Last error message (if any)
    pub last_error: Option<String>,
}

impl PendingTruthSubmission {
    /// Create a new pending submission for the given truth.
    pub fn new(truth: BehavioralTruth) -> Self {
        Self {
            truth,
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
pub struct TruthFeedback {
    /// ID of the truth being reported on
    pub truth_id: String,

    /// Whether this is a reinforcement (true) or contradiction (false)
    pub is_reinforcement: bool,

    /// Optional context about the feedback
    pub context: Option<String>,

    /// When this feedback was generated
    pub timestamp: i64,
}

impl TruthFeedback {
    /// Create a reinforcement feedback for the given truth.
    pub fn reinforcement(truth_id: String, context: Option<String>) -> Self {
        Self {
            truth_id,
            is_reinforcement: true,
            context,
            timestamp: Utc::now().timestamp(),
        }
    }

    /// Create a contradiction feedback for the given truth.
    pub fn contradiction(truth_id: String, context: Option<String>) -> Self {
        Self {
            truth_id,
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
    fn test_truth_creation() {
        let truth = BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "pm2 logs".to_string(),
            "Use --nostream flag to avoid blocking".to_string(),
            "pm2 logs streams indefinitely by default".to_string(),
            TruthSource::ExplicitCommand,
            Some("user123".to_string()),
        );

        assert_eq!(truth.category, TruthCategory::CommandUsage);
        assert_eq!(truth.confidence, 0.8); // ExplicitCommand initial
        assert_eq!(truth.reinforcements, 0);
        assert!(!truth.deleted);
    }

    #[test]
    fn test_reinforcement() {
        let mut truth = BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "test".to_string(),
            "test rule".to_string(),
            "test rationale".to_string(),
            TruthSource::FailurePattern, // 0.5 initial
            None,
        );

        assert_eq!(truth.confidence, 0.5);

        // Reinforce with α = 0.1
        truth.reinforce(0.1);

        // new = 0.1 * 1.0 + 0.9 * 0.5 = 0.1 + 0.45 = 0.55
        assert!((truth.confidence - 0.55).abs() < 0.001);
        assert_eq!(truth.reinforcements, 1);
    }

    #[test]
    fn test_contradiction() {
        let mut truth = BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "test".to_string(),
            "test rule".to_string(),
            "test rationale".to_string(),
            TruthSource::ExplicitCommand, // 0.8 initial
            None,
        );

        // Contradict with α = 0.1
        truth.contradict(0.1);

        // new = 0.1 * 0.0 + 0.9 * 0.8 = 0 + 0.72 = 0.72
        assert!((truth.confidence - 0.72).abs() < 0.001);
        assert_eq!(truth.contradictions, 1);
    }

    #[test]
    fn test_success_rate() {
        let mut truth = BehavioralTruth::new(
            TruthCategory::TaskStrategy,
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
            TruthSource::SuccessPattern,
            None,
        );

        assert_eq!(truth.success_rate(), 0.5); // No data

        truth.reinforcements = 3;
        truth.contradictions = 1;

        assert_eq!(truth.success_rate(), 0.75); // 3/4
    }

    #[test]
    fn test_category_parsing() {
        assert_eq!(
            "cmd".parse::<TruthCategory>().unwrap(),
            TruthCategory::CommandUsage
        );
        assert_eq!(
            "task".parse::<TruthCategory>().unwrap(),
            TruthCategory::TaskStrategy
        );
        assert_eq!(
            "avoid".parse::<TruthCategory>().unwrap(),
            TruthCategory::PatternAvoidance
        );
        assert!("invalid".parse::<TruthCategory>().is_err());
    }

    #[test]
    fn test_source_initial_confidence() {
        assert_eq!(TruthSource::ExplicitCommand.initial_confidence(), 0.8);
        assert_eq!(
            TruthSource::ConversationCorrection.initial_confidence(),
            0.6
        );
        assert_eq!(TruthSource::SuccessPattern.initial_confidence(), 0.4);
        assert_eq!(TruthSource::FailurePattern.initial_confidence(), 0.5);
    }

    #[test]
    fn test_truth_display() {
        let truth = BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "pm2 logs".to_string(),
            "Use --nostream flag".to_string(),
            "Avoids blocking".to_string(),
            TruthSource::ExplicitCommand,
            None,
        );

        let display = format!("{}", truth);
        assert!(display.contains("80%"));
        assert!(display.contains("cmd"));
        assert!(display.contains("Use --nostream flag"));
    }
}
