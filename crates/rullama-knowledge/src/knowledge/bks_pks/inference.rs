//! Truth inference engine
//!
//! Converts patterns and signals into behavioral truths.
//! Handles deduplication, merging, and confidence scoring.

use super::collector::FailurePattern;
use super::truth::{BehavioralTruth, TruthCategory, TruthSource};
use std::collections::HashMap;

/// Engine for inferring truths from patterns
pub struct TruthInferenceEngine {
    /// Known command flags that resolve blocking issues
    known_nonblocking_flags: HashMap<String, String>,

    /// Minimum occurrences to infer a pattern
    min_occurrences: u32,

    /// Client ID for provenance
    client_id: Option<String>,
}

impl TruthInferenceEngine {
    /// Create a new inference engine
    pub fn new(min_occurrences: u32, client_id: Option<String>) -> Self {
        let mut known_flags = HashMap::new();

        // Known fixes for common blocking commands
        known_flags.insert("pm2 logs".to_string(), "--nostream".to_string());
        known_flags.insert("docker logs".to_string(), "--follow=false".to_string());
        known_flags.insert("tail -f".to_string(), "tail -n".to_string());
        known_flags.insert("watch".to_string(), "-n 1 -e".to_string());

        Self {
            known_nonblocking_flags: known_flags,
            min_occurrences,
            client_id,
        }
    }

    /// Infer truth from a failure pattern
    pub fn infer_from_failure(&self, pattern: &FailurePattern) -> Option<BehavioralTruth> {
        if pattern.occurrences < self.min_occurrences {
            return None;
        }

        // Check if we have a known fix
        if let Some(fix) = self.find_known_fix(&pattern.pattern) {
            return Some(self.create_command_fix_truth(pattern, &fix));
        }

        // Check if this looks like a timeout/blocking issue
        if self.looks_like_blocking(&pattern.error_pattern) {
            return Some(self.create_blocking_warning_truth(pattern));
        }

        // Generic failure pattern
        Some(self.create_generic_failure_truth(pattern))
    }

    /// Find a known fix for a command pattern
    fn find_known_fix(&self, pattern: &str) -> Option<String> {
        for (cmd, fix) in &self.known_nonblocking_flags {
            if pattern.to_lowercase().contains(&cmd.to_lowercase()) {
                return Some(fix.clone());
            }
        }
        None
    }

    /// Check if error looks like blocking/timeout
    fn looks_like_blocking(&self, error_pattern: &Option<String>) -> bool {
        if let Some(error) = error_pattern {
            let error_lower = error.to_lowercase();
            error_lower.contains("timeout")
                || error_lower.contains("block")
                || error_lower.contains("hang")
                || error_lower.contains("wait")
                || error_lower.contains("stuck")
        } else {
            false
        }
    }

    /// Create a truth for a known command fix
    fn create_command_fix_truth(&self, pattern: &FailurePattern, fix: &str) -> BehavioralTruth {
        let rule = format!(
            "Use '{}' flag with '{}' to avoid blocking",
            fix, pattern.pattern
        );

        let rationale = format!(
            "'{}' without '{}' can block indefinitely. Detected {} failures.",
            pattern.pattern, fix, pattern.occurrences
        );

        BehavioralTruth::new(
            TruthCategory::CommandUsage,
            pattern.pattern.clone(),
            rule,
            rationale,
            TruthSource::FailurePattern,
            self.client_id.clone(),
        )
    }

    /// Create a truth warning about blocking behavior
    fn create_blocking_warning_truth(&self, pattern: &FailurePattern) -> BehavioralTruth {
        let rule = format!(
            "'{}' may block or timeout - consider using a non-blocking alternative or spawning a monitor",
            pattern.pattern
        );

        let rationale = format!(
            "Detected {} timeout/blocking failures with '{}'",
            pattern.occurrences, pattern.pattern
        );

        BehavioralTruth::new(
            TruthCategory::PatternAvoidance,
            pattern.pattern.clone(),
            rule,
            rationale,
            TruthSource::FailurePattern,
            self.client_id.clone(),
        )
    }

    /// Create a generic failure truth
    fn create_generic_failure_truth(&self, pattern: &FailurePattern) -> BehavioralTruth {
        let error_info = pattern
            .error_pattern
            .as_ref()
            .map(|e| format!(" (error: {})", truncate(e, 50)))
            .unwrap_or_default();

        let rule = format!(
            "'{}' frequently fails{} - consider alternatives",
            pattern.pattern, error_info
        );

        let rationale = format!(
            "Detected {} failures with '{}' across {} contexts",
            pattern.occurrences,
            pattern.pattern,
            pattern.contexts.len()
        );

        BehavioralTruth::new(
            TruthCategory::PatternAvoidance,
            pattern.pattern.clone(),
            rule,
            rationale,
            TruthSource::FailurePattern,
            self.client_id.clone(),
        )
    }

    /// Infer category from correction context
    pub fn infer_category_from_correction(
        &self,
        context: &str,
        wrong: &str,
        right: &str,
    ) -> TruthCategory {
        let combined = format!("{} {} {}", context, wrong, right).to_lowercase();

        // Check for specific patterns
        if combined.contains("spawn") || combined.contains("agent") || combined.contains("monitor")
        {
            TruthCategory::TaskStrategy
        } else if combined.contains("--") || combined.contains("flag") {
            TruthCategory::CommandUsage
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
        } else {
            TruthCategory::ToolBehavior
        }
    }

    /// Create a truth from a correction
    pub fn create_correction_truth(
        &self,
        context: &str,
        wrong: &str,
        right: &str,
    ) -> BehavioralTruth {
        let category = self.infer_category_from_correction(context, wrong, right);

        let rule = format!(
            "Instead of '{}', use '{}'",
            truncate(wrong, 50),
            truncate(right, 50)
        );

        let rationale = format!(
            "User corrected behavior in context: {}",
            truncate(context, 100)
        );

        BehavioralTruth::new(
            category,
            context.to_string(),
            rule,
            rationale,
            TruthSource::ConversationCorrection,
            self.client_id.clone(),
        )
    }

    /// Create a truth from explicit teaching
    pub fn create_explicit_truth(
        &self,
        rule: &str,
        rationale: Option<&str>,
        category: TruthCategory,
        context: Option<&str>,
    ) -> BehavioralTruth {
        let context_pattern = context
            .map(|c| c.to_string())
            .unwrap_or_else(|| extract_context_from_rule(rule));

        let rationale = rationale
            .map(|r| r.to_string())
            .unwrap_or_else(|| "Explicitly taught by user".to_string());

        BehavioralTruth::new(
            category,
            context_pattern,
            rule.to_string(),
            rationale,
            TruthSource::ExplicitCommand,
            self.client_id.clone(),
        )
    }

    /// Check if two truths are similar enough to merge
    pub fn should_merge(&self, existing: &BehavioralTruth, new: &BehavioralTruth) -> bool {
        // Same category
        if existing.category != new.category {
            return false;
        }

        // Similar context patterns
        let context_similarity = jaccard_similarity(
            &existing.context_pattern.to_lowercase(),
            &new.context_pattern.to_lowercase(),
        );

        // Similar rules
        let rule_similarity =
            jaccard_similarity(&existing.rule.to_lowercase(), &new.rule.to_lowercase());

        context_similarity > 0.5 && rule_similarity > 0.3
    }

    /// Merge a new truth into an existing one
    pub fn merge_truths(&self, existing: &mut BehavioralTruth, new: &BehavioralTruth) {
        // Combine reinforcements
        existing.reinforcements += new.reinforcements;
        existing.contradictions += new.contradictions;

        // Average confidence with bias toward newer
        existing.confidence = 0.7 * existing.confidence + 0.3 * new.confidence;

        // Update timestamp
        if new.last_used > existing.last_used {
            existing.last_used = new.last_used;
        }

        existing.version += 1;
    }
}

/// Extract context from a rule string
fn extract_context_from_rule(rule: &str) -> String {
    // Look for quoted strings first
    if let Some(start) = rule.find('\'')
        && let Some(end) = rule[start + 1..].find('\'')
    {
        return rule[start + 1..start + 1 + end].to_string();
    }

    // Look for command-like patterns
    let words: Vec<&str> = rule.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        // Skip common words
        if [
            "use", "with", "the", "a", "to", "for", "when", "if", "instead", "of",
        ]
        .contains(&word.to_lowercase().as_str())
        {
            continue;
        }

        // Found a potential command
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

/// Calculate Jaccard similarity between two strings (word-based)
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Truncate a string
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_from_known_failure() {
        let engine = TruthInferenceEngine::new(3, None);

        let pattern = FailurePattern {
            pattern: "pm2 logs myapp".to_string(),
            error_pattern: Some("timeout".to_string()),
            occurrences: 5,
            timestamps: vec![1, 2, 3, 4, 5],
            contexts: vec!["test".to_string()],
        };

        let truth = engine.infer_from_failure(&pattern).unwrap();
        assert!(truth.rule.contains("--nostream"));
        assert_eq!(truth.category, TruthCategory::CommandUsage);
    }

    #[test]
    fn test_infer_from_blocking_failure() {
        let engine = TruthInferenceEngine::new(3, None);

        let pattern = FailurePattern {
            pattern: "some-command".to_string(),
            error_pattern: Some("connection timeout after 30s".to_string()),
            occurrences: 3,
            timestamps: vec![1, 2, 3],
            contexts: vec!["test".to_string()],
        };

        let truth = engine.infer_from_failure(&pattern).unwrap();
        assert!(truth.rule.contains("block") || truth.rule.contains("timeout"));
    }

    #[test]
    fn test_category_inference() {
        let engine = TruthInferenceEngine::new(3, None);

        assert_eq!(
            engine.infer_category_from_correction("task", "poll inline", "spawn agent"),
            TruthCategory::TaskStrategy
        );

        assert_eq!(
            engine.infer_category_from_correction("pm2", "logs", "--nostream flag"),
            TruthCategory::CommandUsage
        );
    }

    #[test]
    fn test_jaccard_similarity() {
        assert_eq!(jaccard_similarity("a b c", "a b c"), 1.0);
        assert_eq!(jaccard_similarity("a b c", "d e f"), 0.0);
        assert!((jaccard_similarity("a b c", "a b d") - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extract_context() {
        assert_eq!(
            extract_context_from_rule("Use '--nostream' with pm2 logs"),
            "--nostream"
        );
        assert_eq!(
            extract_context_from_rule("cargo build should use cargo-watch"),
            "cargo build"
        );
    }

    #[test]
    fn test_should_merge() {
        let engine = TruthInferenceEngine::new(3, None);

        let truth1 = BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "pm2 logs".to_string(),
            "Use --nostream flag".to_string(),
            "Avoids blocking".to_string(),
            TruthSource::ExplicitCommand,
            None,
        );

        let truth2 = BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "pm2 logs app".to_string(),
            "Use --nostream flag to avoid blocking".to_string(),
            "Different rationale".to_string(),
            TruthSource::FailurePattern,
            None,
        );

        assert!(engine.should_merge(&truth1, &truth2));

        let truth3 = BehavioralTruth::new(
            TruthCategory::TaskStrategy,
            "something else".to_string(),
            "Different rule entirely".to_string(),
            "Different".to_string(),
            TruthSource::ExplicitCommand,
            None,
        );

        assert!(!engine.should_merge(&truth1, &truth3));
    }
}
