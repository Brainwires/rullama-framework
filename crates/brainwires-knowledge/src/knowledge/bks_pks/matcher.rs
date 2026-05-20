//! Context matcher for truth retrieval
//!
//! Matches current context against stored truths to determine which
//! truths are relevant for prompt injection.

use super::truth::{BehavioralTruth, TruthCategory};
use std::collections::HashSet;

/// Context matcher for finding relevant truths
pub struct ContextMatcher {
    /// Minimum confidence for a truth to be considered
    min_confidence: f32,

    /// Decay days for confidence calculation
    decay_days: u32,

    /// Maximum truths to return
    max_results: usize,

    /// Words to ignore in matching
    stop_words: HashSet<String>,
}

impl ContextMatcher {
    /// Create a new context matcher
    pub fn new(min_confidence: f32, decay_days: u32, max_results: usize) -> Self {
        let stop_words: HashSet<String> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below", "between",
            "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just", "also", "now",
            "and", "but", "or", "if", "because", "until", "while", "this", "that", "these",
            "those", "it", "its",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            min_confidence,
            decay_days,
            max_results,
            stop_words,
        }
    }

    /// Find truths matching the given context
    pub fn find_matches<'a>(
        &self,
        context: &str,
        truths: impl Iterator<Item = &'a BehavioralTruth>,
    ) -> Vec<MatchedTruth<'a>> {
        let context_words = self.tokenize(context);

        let mut matches: Vec<MatchedTruth> = truths
            .filter(|t| !t.deleted && t.is_reliable(self.min_confidence, self.decay_days))
            .filter_map(|truth| {
                let score = self.calculate_match_score(&context_words, truth);
                if score > 0.0 {
                    Some(MatchedTruth {
                        truth,
                        match_score: score,
                        effective_confidence: truth.decayed_confidence(self.decay_days),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by combined score (match_score * effective_confidence)
        matches.sort_by(|a, b| {
            let score_a = a.combined_score();
            let score_b = b.combined_score();
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches.truncate(self.max_results);
        matches
    }

    /// Find truths by category
    pub fn find_by_category<'a>(
        &self,
        category: TruthCategory,
        truths: impl Iterator<Item = &'a BehavioralTruth>,
    ) -> Vec<&'a BehavioralTruth> {
        truths
            .filter(|t| {
                !t.deleted
                    && t.category == category
                    && t.is_reliable(self.min_confidence, self.decay_days)
            })
            .collect()
    }

    /// Search truths by keyword
    pub fn search<'a>(
        &self,
        query: &str,
        truths: impl Iterator<Item = &'a BehavioralTruth>,
    ) -> Vec<MatchedTruth<'a>> {
        let query_words = self.tokenize(query);

        let mut matches: Vec<MatchedTruth> = truths
            .filter(|t| !t.deleted)
            .filter_map(|truth| {
                let score = self.calculate_search_score(&query_words, truth);
                if score > 0.0 {
                    Some(MatchedTruth {
                        truth,
                        match_score: score,
                        effective_confidence: truth.decayed_confidence(self.decay_days),
                    })
                } else {
                    None
                }
            })
            .collect();

        matches.sort_by(|a, b| {
            b.match_score
                .partial_cmp(&a.match_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches.truncate(self.max_results);
        matches
    }

    /// Tokenize text into words (lowercase, no stop words)
    fn tokenize(&self, text: &str) -> HashSet<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|w| !w.is_empty() && w.len() > 1 && !self.stop_words.contains(*w))
            .map(|w| w.to_string())
            .collect()
    }

    /// Calculate match score for context matching
    fn calculate_match_score(
        &self,
        context_words: &HashSet<String>,
        truth: &BehavioralTruth,
    ) -> f64 {
        let pattern_words = self.tokenize(&truth.context_pattern);

        if pattern_words.is_empty() {
            return 0.0;
        }

        // Calculate how many pattern words are in context
        let matches = pattern_words.intersection(context_words).count();

        if matches == 0 {
            return 0.0;
        }

        // Score is based on coverage of pattern
        let coverage = matches as f64 / pattern_words.len() as f64;

        // Boost if context contains all pattern words
        if matches == pattern_words.len() {
            coverage * 1.5
        } else {
            coverage
        }
    }

    /// Calculate match score for search
    fn calculate_search_score(
        &self,
        query_words: &HashSet<String>,
        truth: &BehavioralTruth,
    ) -> f64 {
        if query_words.is_empty() {
            return 0.0;
        }

        let pattern_words = self.tokenize(&truth.context_pattern);
        let rule_words = self.tokenize(&truth.rule);
        let rationale_words = self.tokenize(&truth.rationale);

        let all_words: HashSet<_> = pattern_words
            .union(&rule_words)
            .cloned()
            .collect::<HashSet<_>>()
            .union(&rationale_words)
            .cloned()
            .collect();

        // Check for exact word matches
        let word_matches = query_words.intersection(&all_words).count();

        // Also check for substring matches in the original text
        let combined_text = format!(
            "{} {} {}",
            truth.context_pattern.to_lowercase(),
            truth.rule.to_lowercase(),
            truth.rationale.to_lowercase()
        );

        let substring_matches = query_words
            .iter()
            .filter(|q| combined_text.contains(q.as_str()))
            .count();

        let total_matches = word_matches.max(substring_matches);

        if total_matches == 0 {
            return 0.0;
        }

        // Score based on query coverage
        total_matches as f64 / query_words.len() as f64
    }

    /// Check if a truth conflicts with user instruction
    pub fn detect_conflict(
        &self,
        instruction: &str,
        truth: &BehavioralTruth,
    ) -> Option<ConflictInfo> {
        let instruction_lower = instruction.to_lowercase();
        let pattern_lower = truth.context_pattern.to_lowercase();

        // Check if instruction mentions the truth's context
        let context_match = pattern_lower
            .split_whitespace()
            .any(|word| instruction_lower.contains(word));

        if !context_match {
            return None;
        }

        // Check for explicit contradictions
        let instruction_words: HashSet<_> = instruction_lower.split_whitespace().collect();

        // If truth says "use X" but instruction doesn't mention X
        if truth.rule.to_lowercase().contains("use ") {
            // Extract what to use
            if let Some(idx) = truth.rule.to_lowercase().find("use ") {
                let suggested = &truth.rule[idx + 4..];
                let suggested_word = suggested.split_whitespace().next().unwrap_or("");

                // Check if instruction explicitly avoids this
                if !instruction_words.contains(suggested_word) {
                    return Some(ConflictInfo {
                        truth_id: truth.id.clone(),
                        conflict_type: ConflictType::MissingSuggested,
                        suggested_action: format!(
                            "Add {} as suggested by learned rule",
                            suggested_word
                        ),
                        confidence: truth.decayed_confidence(self.decay_days),
                    });
                }
            }
        }

        // If truth says "avoid X" but instruction uses X
        if truth.rule.to_lowercase().contains("avoid ")
            || truth.rule.to_lowercase().contains("don't ")
        {
            // This might be a conflict
            return Some(ConflictInfo {
                truth_id: truth.id.clone(),
                conflict_type: ConflictType::UsingAvoided,
                suggested_action: format!("Consider: {}", truth.rule),
                confidence: truth.decayed_confidence(self.decay_days),
            });
        }

        None
    }
}

/// A truth matched to a context
#[derive(Debug, Clone)]
pub struct MatchedTruth<'a> {
    /// The matched truth
    pub truth: &'a BehavioralTruth,

    /// How well this truth matches the context (0.0 - 1.0+)
    pub match_score: f64,

    /// Confidence after decay
    pub effective_confidence: f32,
}

impl<'a> MatchedTruth<'a> {
    /// Combined score for ranking
    pub fn combined_score(&self) -> f64 {
        self.match_score * self.effective_confidence as f64
    }
}

/// Information about a conflict between truth and instruction
#[derive(Debug, Clone)]
pub struct ConflictInfo {
    /// ID of the conflicting truth
    pub truth_id: String,

    /// Type of conflict
    pub conflict_type: ConflictType,

    /// Suggested action to resolve
    pub suggested_action: String,

    /// Confidence in the truth
    pub confidence: f32,
}

/// Type of conflict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Instruction is missing something the truth suggests
    MissingSuggested,

    /// Instruction uses something the truth says to avoid
    UsingAvoided,

    /// General conflict
    General,
}

/// Format truths for prompt injection
pub fn format_truths_for_prompt(matches: &[MatchedTruth]) -> String {
    if matches.is_empty() {
        return String::new();
    }

    let mut output = String::from("\n## Learned Behavioral Knowledge\n\n");
    output.push_str("The following are learned behaviors from collective experience:\n\n");

    for matched in matches {
        output.push_str(&format!(
            "- **[{:.0}%]** {}: {}\n",
            matched.effective_confidence * 100.0,
            matched.truth.category,
            matched.truth.rule
        ));

        if !matched.truth.rationale.is_empty() {
            output.push_str(&format!("  _Reason: {}_\n", matched.truth.rationale));
        }
    }

    output.push('\n');
    output
}

/// Format a conflict prompt for user clarification
pub fn format_conflict_prompt(truth: &BehavioralTruth, conflict: &ConflictInfo) -> String {
    format!(
        "I've learned that {} (confidence: {:.0}%). {}.\n\nShould I:\n1. Follow the learned behavior (recommended)\n2. Proceed as you specified",
        truth.rule,
        conflict.confidence * 100.0,
        truth.rationale
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::bks_pks::truth::TruthSource;

    fn create_test_truth(context: &str, rule: &str) -> BehavioralTruth {
        BehavioralTruth::new(
            TruthCategory::CommandUsage,
            context.to_string(),
            rule.to_string(),
            "Test rationale".to_string(),
            TruthSource::ExplicitCommand,
            None,
        )
    }

    #[test]
    fn test_find_matches() {
        let matcher = ContextMatcher::new(0.5, 30, 10);

        let truths = [
            create_test_truth("pm2 logs", "Use --nostream flag"),
            create_test_truth("cargo build", "Use cargo-watch for watch mode"),
        ];

        let matches = matcher.find_matches("run pm2 logs for my app", truths.iter());

        assert_eq!(matches.len(), 1);
        assert!(matches[0].truth.rule.contains("--nostream"));
    }

    #[test]
    fn test_search() {
        let matcher = ContextMatcher::new(0.0, 30, 10);

        let truths = [
            create_test_truth("pm2 logs", "Use --nostream flag to avoid blocking"),
            create_test_truth("docker", "Use --follow=false"),
        ];

        let matches = matcher.search("nostream", truths.iter());

        assert_eq!(matches.len(), 1);
        assert!(matches[0].truth.rule.contains("--nostream"));
    }

    #[test]
    fn test_tokenize() {
        let matcher = ContextMatcher::new(0.5, 30, 10);

        let tokens = matcher.tokenize("pm2 logs --nostream");
        assert!(tokens.contains("pm2"));
        assert!(tokens.contains("logs"));
        assert!(tokens.contains("--nostream"));
        assert!(!tokens.contains("the")); // Stop word
    }

    #[test]
    fn test_format_truths_for_prompt() {
        let truth = create_test_truth("pm2 logs", "Use --nostream flag");
        let matched = MatchedTruth {
            truth: &truth,
            match_score: 0.8,
            effective_confidence: 0.9,
        };

        let output = format_truths_for_prompt(&[matched]);

        assert!(output.contains("Learned Behavioral Knowledge"));
        assert!(output.contains("--nostream"));
        assert!(output.contains("90%"));
    }

    #[test]
    fn test_conflict_detection() {
        let matcher = ContextMatcher::new(0.5, 30, 10);

        let truth = create_test_truth("pm2 logs", "Use --nostream flag");

        let conflict = matcher.detect_conflict("show me pm2 logs", &truth);
        assert!(conflict.is_some());
    }
}
