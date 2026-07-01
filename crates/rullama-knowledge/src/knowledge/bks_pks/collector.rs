//! Learning signal collector
//!
//! Collects learning signals from all three sources:
//! 1. Explicit: /learn command
//! 2. Implicit: Conversation corrections
//! 3. Aggressive: Success/failure patterns

use super::truth::{BehavioralTruth, TruthCategory, TruthSource};
use chrono::Utc;
use std::collections::HashMap;

/// A learning signal that may result in a new truth
#[derive(Debug, Clone)]
pub enum LearningSignal {
    /// User explicitly taught something via /learn command
    ExplicitTeaching {
        /// The rule being taught.
        rule: String,
        /// Rationale for the rule.
        rationale: Option<String>,
        /// Category of the truth.
        category: TruthCategory,
        /// Context where the rule applies.
        context: Option<String>,
    },

    /// User corrected the agent in conversation
    Correction {
        /// Context of the correction.
        context: String,
        /// What the agent did wrong.
        wrong_behavior: String,
        /// What the agent should have done.
        right_behavior: String,
    },

    /// Tool execution outcome
    ToolOutcome {
        /// Name of the tool executed.
        tool_name: String,
        /// Command or arguments used.
        command: String,
        /// Whether execution succeeded.
        success: bool,
        /// Error message if failed.
        error_message: Option<String>,
        /// Execution time in milliseconds.
        execution_time_ms: u64,
    },

    /// Strategy outcome (how a task was approached)
    StrategyOutcome {
        /// Description of the strategy used.
        strategy: String,
        /// Context of the task.
        context: String,
        /// Whether the strategy succeeded.
        success: bool,
        /// Additional details.
        details: Option<String>,
    },
}

/// Tracks failure patterns to detect anti-patterns
#[derive(Debug, Clone)]
pub struct FailurePattern {
    /// Command pattern (e.g., "pm2 logs")
    pub pattern: String,

    /// Error message pattern
    pub error_pattern: Option<String>,

    /// Number of occurrences
    pub occurrences: u32,

    /// Timestamps of occurrences
    pub timestamps: Vec<i64>,

    /// Associated context
    pub contexts: Vec<String>,
}

impl FailurePattern {
    /// Create a new failure pattern with its first occurrence.
    pub fn new(pattern: String, error_pattern: Option<String>, context: String) -> Self {
        Self {
            pattern,
            error_pattern,
            occurrences: 1,
            timestamps: vec![Utc::now().timestamp()],
            contexts: vec![context],
        }
    }

    /// Record an additional occurrence of this failure pattern.
    pub fn record_occurrence(&mut self, context: String) {
        self.occurrences += 1;
        self.timestamps.push(Utc::now().timestamp());
        if !self.contexts.contains(&context) {
            self.contexts.push(context);
        }
    }

    /// Check if pattern occurs frequently enough to learn from
    pub fn is_significant(&self, threshold: u32) -> bool {
        self.occurrences >= threshold
    }
}

/// Collects learning signals from all sources
pub struct LearningCollector {
    /// Queued signals waiting to be processed
    signals: Vec<LearningSignal>,

    /// Failure patterns being tracked
    failure_patterns: HashMap<String, FailurePattern>,

    /// Success patterns being tracked
    success_patterns: HashMap<String, u32>,

    /// Threshold for pattern detection
    failure_threshold: u32,

    /// Client ID for provenance
    client_id: Option<String>,
}

impl LearningCollector {
    /// Create a new collector
    pub fn new(failure_threshold: u32, client_id: Option<String>) -> Self {
        Self {
            signals: Vec::new(),
            failure_patterns: HashMap::new(),
            success_patterns: HashMap::new(),
            failure_threshold,
            client_id,
        }
    }

    /// Record explicit teaching from /learn command
    pub fn record_explicit_teaching(
        &mut self,
        rule: &str,
        rationale: Option<&str>,
        category: TruthCategory,
        context: Option<&str>,
    ) {
        self.signals.push(LearningSignal::ExplicitTeaching {
            rule: rule.to_string(),
            rationale: rationale.map(|s| s.to_string()),
            category,
            context: context.map(|s| s.to_string()),
        });
    }

    /// Record a correction from conversation
    pub fn record_correction(&mut self, context: &str, wrong_behavior: &str, right_behavior: &str) {
        self.signals.push(LearningSignal::Correction {
            context: context.to_string(),
            wrong_behavior: wrong_behavior.to_string(),
            right_behavior: right_behavior.to_string(),
        });
    }

    /// Record a tool execution outcome
    pub fn record_tool_outcome(
        &mut self,
        tool_name: &str,
        command: &str,
        success: bool,
        error_message: Option<&str>,
        execution_time_ms: u64,
    ) {
        // Track pattern for aggressive learning
        let pattern_key = Self::extract_command_pattern(command);

        if success {
            *self.success_patterns.entry(pattern_key).or_insert(0) += 1;
        } else {
            let error_pattern = error_message.map(Self::extract_error_pattern);

            if let Some(existing) = self.failure_patterns.get_mut(&pattern_key) {
                existing.record_occurrence(command.to_string());
            } else {
                self.failure_patterns.insert(
                    pattern_key.clone(),
                    FailurePattern::new(pattern_key, error_pattern, command.to_string()),
                );
            }
        }

        self.signals.push(LearningSignal::ToolOutcome {
            tool_name: tool_name.to_string(),
            command: command.to_string(),
            success,
            error_message: error_message.map(|s| s.to_string()),
            execution_time_ms,
        });
    }

    /// Record a strategy outcome
    pub fn record_strategy_outcome(
        &mut self,
        strategy: &str,
        context: &str,
        success: bool,
        details: Option<&str>,
    ) {
        self.signals.push(LearningSignal::StrategyOutcome {
            strategy: strategy.to_string(),
            context: context.to_string(),
            success,
            details: details.map(|s| s.to_string()),
        });
    }

    /// Extract command pattern from full command
    /// e.g., "pm2 logs myapp --lines 100" → "pm2 logs"
    fn extract_command_pattern(command: &str) -> String {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.len() >= 2 {
            format!("{} {}", parts[0], parts[1])
        } else if parts.len() == 1 {
            parts[0].to_string()
        } else {
            command.to_string()
        }
    }

    /// Extract error pattern from error message
    /// Normalizes variable parts like paths, numbers, etc.
    fn extract_error_pattern(error: &str) -> String {
        // Simple normalization - could be more sophisticated
        let normalized = error
            .chars()
            .take(100) // Limit length
            .collect::<String>()
            .to_lowercase();

        // Remove numbers and paths
        use std::sync::LazyLock;
        static RE_NUMBERS: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"\d+").expect("valid regex"));
        static RE_PATHS: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"/[\w/.-]+").expect("valid regex"));
        let re_numbers = &*RE_NUMBERS;
        let re_paths = &*RE_PATHS;

        let normalized = re_numbers.replace_all(&normalized, "<N>");
        let normalized = re_paths.replace_all(&normalized, "<PATH>");

        normalized.to_string()
    }

    /// Get significant failure patterns (above threshold)
    pub fn get_significant_failures(&self) -> Vec<&FailurePattern> {
        self.failure_patterns
            .values()
            .filter(|p| p.is_significant(self.failure_threshold))
            .collect()
    }

    /// Take all queued signals
    pub fn take_signals(&mut self) -> Vec<LearningSignal> {
        std::mem::take(&mut self.signals)
    }

    /// Process signals and generate truths
    pub fn process_signals(&mut self) -> Vec<BehavioralTruth> {
        let signals = self.take_signals();
        let mut truths = Vec::new();

        for signal in signals {
            if let Some(truth) = self.signal_to_truth(signal) {
                truths.push(truth);
            }
        }

        // Also check for significant failure patterns
        for pattern in self.get_significant_failures() {
            if let Some(truth) = self.failure_pattern_to_truth(pattern) {
                truths.push(truth);
            }
        }

        truths
    }

    /// Convert a signal to a truth (if applicable)
    fn signal_to_truth(&self, signal: LearningSignal) -> Option<BehavioralTruth> {
        match signal {
            LearningSignal::ExplicitTeaching {
                rule,
                rationale,
                category,
                context,
            } => {
                let context_pattern = context.unwrap_or_else(|| {
                    // Extract context from rule if not provided
                    Self::extract_context_from_rule(&rule)
                });

                Some(BehavioralTruth::new(
                    category,
                    context_pattern,
                    rule,
                    rationale.unwrap_or_else(|| "Explicitly taught by user".to_string()),
                    TruthSource::ExplicitCommand,
                    self.client_id.clone(),
                ))
            }

            LearningSignal::Correction {
                context,
                wrong_behavior,
                right_behavior,
            } => {
                let rule = format!(
                    "Instead of '{}', use '{}'",
                    truncate(&wrong_behavior, 50),
                    truncate(&right_behavior, 50)
                );

                let rationale = format!(
                    "User corrected agent behavior from '{}' to '{}'",
                    truncate(&wrong_behavior, 30),
                    truncate(&right_behavior, 30)
                );

                let category = Self::infer_category(&context, &right_behavior);

                Some(BehavioralTruth::new(
                    category,
                    context,
                    rule,
                    rationale,
                    TruthSource::ConversationCorrection,
                    self.client_id.clone(),
                ))
            }

            LearningSignal::ToolOutcome { .. } => {
                // Tool outcomes are aggregated into patterns, not individual truths
                None
            }

            LearningSignal::StrategyOutcome {
                strategy,
                context,
                success,
                details,
            } => {
                if success {
                    Some(BehavioralTruth::new(
                        TruthCategory::TaskStrategy,
                        context,
                        format!("Use strategy: {}", strategy),
                        details.unwrap_or_else(|| "Successful strategy execution".to_string()),
                        TruthSource::SuccessPattern,
                        self.client_id.clone(),
                    ))
                } else {
                    None // Failures go into patterns
                }
            }
        }
    }

    /// Convert a failure pattern to a truth
    fn failure_pattern_to_truth(&self, pattern: &FailurePattern) -> Option<BehavioralTruth> {
        // Generate a rule based on the failure pattern
        let rule = if let Some(ref error) = pattern.error_pattern {
            format!(
                "Avoid '{}' - causes error: {}",
                pattern.pattern,
                truncate(error, 50)
            )
        } else {
            format!(
                "'{}' frequently fails ({} times)",
                pattern.pattern, pattern.occurrences
            )
        };

        let rationale = format!(
            "Detected {} failures across {} contexts",
            pattern.occurrences,
            pattern.contexts.len()
        );

        Some(BehavioralTruth::new(
            TruthCategory::PatternAvoidance,
            pattern.pattern.clone(),
            rule,
            rationale,
            TruthSource::FailurePattern,
            self.client_id.clone(),
        ))
    }

    /// Extract context from a rule string
    fn extract_context_from_rule(rule: &str) -> String {
        // Look for command-like patterns
        let words: Vec<&str> = rule.split_whitespace().collect();

        // Find first word that looks like a command
        for (i, word) in words.iter().enumerate() {
            if word
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            {
                if i + 1 < words.len() {
                    return format!("{} {}", word, words[i + 1]);
                }
                return word.to_string();
            }
        }

        // Fallback: first few words
        words.iter().take(3).cloned().collect::<Vec<_>>().join(" ")
    }

    /// Infer category from context and behavior
    fn infer_category(context: &str, behavior: &str) -> TruthCategory {
        let combined = format!("{} {}", context, behavior).to_lowercase();

        if combined.contains("spawn") || combined.contains("agent") || combined.contains("monitor")
        {
            TruthCategory::TaskStrategy
        } else if combined.contains("error")
            || combined.contains("fail")
            || combined.contains("retry")
        {
            TruthCategory::ErrorRecovery
        } else if combined.contains("context")
            || combined.contains("token")
            || combined.contains("parallel")
        {
            TruthCategory::ResourceManagement
        } else if combined.contains("don't")
            || combined.contains("avoid")
            || combined.contains("never")
        {
            TruthCategory::PatternAvoidance
        } else if combined.contains("--")
            || combined.contains("flag")
            || combined.contains("option")
        {
            TruthCategory::CommandUsage
        } else {
            TruthCategory::ToolBehavior
        }
    }

    /// Clear all tracked patterns
    pub fn clear_patterns(&mut self) {
        self.failure_patterns.clear();
        self.success_patterns.clear();
    }

    /// Get statistics about collected signals
    pub fn stats(&self) -> CollectorStats {
        CollectorStats {
            pending_signals: self.signals.len(),
            tracked_failure_patterns: self.failure_patterns.len(),
            tracked_success_patterns: self.success_patterns.len(),
            significant_failures: self.get_significant_failures().len(),
        }
    }
}

/// Statistics about the collector
#[derive(Debug, Clone)]
pub struct CollectorStats {
    /// Number of unprocessed learning signals.
    pub pending_signals: usize,
    /// Number of tracked failure patterns.
    pub tracked_failure_patterns: usize,
    /// Number of tracked success patterns.
    pub tracked_success_patterns: usize,
    /// Number of failures that crossed the significance threshold.
    pub significant_failures: usize,
}

/// Truncate a string to a maximum length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Detect corrections in user messages
///
/// Looks for patterns like:
/// - "no, use X instead"
/// - "don't do X, do Y"
/// - "instead of X, use Y"
/// - "X doesn't work, use Y"
pub fn detect_correction(message: &str) -> Option<(String, String)> {
    let message_lower = message.to_lowercase();

    // Pattern: "no, use X instead" or "instead, use X"
    if let Some(idx) = message_lower.find("instead") {
        let before = &message[..idx];
        let after = &message[idx..];

        // Extract the "wrong" from before and "right" from after
        if let Some(use_idx) = after.to_lowercase().find("use ") {
            let right = after[use_idx + 4..]
                .split_whitespace()
                .take(5)
                .collect::<Vec<_>>()
                .join(" ");

            let wrong = before
                .split_whitespace()
                .rev()
                .take(5)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join(" ");

            if !wrong.is_empty() && !right.is_empty() {
                return Some((wrong, right));
            }
        }
    }

    // Pattern: "don't X, do Y" or "don't X, use Y"
    if message_lower.contains("don't") || message_lower.contains("do not") {
        let parts: Vec<&str> = message.split(',').collect();
        if parts.len() >= 2 {
            let wrong = parts[0]
                .to_lowercase()
                .replace("don't", "")
                .replace("do not", "")
                .trim()
                .to_string();

            let right = parts[1].trim().to_string();

            if !wrong.is_empty() && !right.is_empty() {
                return Some((wrong, right));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_teaching() {
        let mut collector = LearningCollector::new(3, None);

        collector.record_explicit_teaching(
            "Use --nostream with pm2 logs",
            Some("Avoids blocking"),
            TruthCategory::CommandUsage,
            Some("pm2 logs"),
        );

        let truths = collector.process_signals();
        assert_eq!(truths.len(), 1);
        assert_eq!(truths[0].source, TruthSource::ExplicitCommand);
        assert!(truths[0].rule.contains("--nostream"));
    }

    #[test]
    fn test_correction() {
        let mut collector = LearningCollector::new(3, None);

        collector.record_correction(
            "long-running task monitoring",
            "inline polling with 300s interval",
            "spawn dedicated monitoring agent",
        );

        let truths = collector.process_signals();
        assert_eq!(truths.len(), 1);
        assert_eq!(truths[0].source, TruthSource::ConversationCorrection);
    }

    #[test]
    fn test_failure_pattern_detection() {
        let mut collector = LearningCollector::new(3, None);

        // Record 3 failures
        for i in 0..3 {
            collector.record_tool_outcome(
                "bash",
                &format!("pm2 logs app{}", i),
                false,
                Some("timeout"),
                30000,
            );
        }

        let significant = collector.get_significant_failures();
        assert_eq!(significant.len(), 1);
        assert!(significant[0].pattern.contains("pm2 logs"));
    }

    #[test]
    fn test_extract_command_pattern() {
        assert_eq!(
            LearningCollector::extract_command_pattern("pm2 logs myapp --lines 100"),
            "pm2 logs"
        );
        assert_eq!(
            LearningCollector::extract_command_pattern("cargo build"),
            "cargo build"
        );
        assert_eq!(LearningCollector::extract_command_pattern("ls"), "ls");
    }

    #[test]
    fn test_detect_correction_instead() {
        let result = detect_correction("No, instead use --nostream flag");
        assert!(result.is_some());
        let (_wrong, right) = result.unwrap();
        assert!(right.contains("--nostream"));
    }

    #[test]
    fn test_detect_correction_dont() {
        let result = detect_correction("Don't poll inline, spawn a monitor agent");
        assert!(result.is_some());
        let (wrong, right) = result.unwrap();
        assert!(wrong.contains("poll"));
        assert!(right.contains("spawn"));
    }

    #[test]
    fn test_infer_category() {
        assert_eq!(
            LearningCollector::infer_category("task", "spawn a monitoring agent"),
            TruthCategory::TaskStrategy
        );
        assert_eq!(
            LearningCollector::infer_category("pm2", "use --nostream flag"),
            TruthCategory::CommandUsage
        );
        assert_eq!(
            LearningCollector::infer_category("api", "retry on timeout"),
            TruthCategory::ErrorRecovery
        );
    }
}
