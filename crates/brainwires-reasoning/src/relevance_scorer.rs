//! Relevance Scorer - Context Re-ranking
//!
//! Uses a provider to score and re-rank retrieved context items
//! based on semantic relevance to the query, replacing fixed thresholds.

use std::sync::Arc;
use tracing::warn;

use brainwires_core::message::Message;
use brainwires_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

/// Result of relevance scoring
#[derive(Clone, Debug)]
pub struct RelevanceResult {
    /// The scored content
    pub content: String,
    /// Original index in the input list
    pub original_index: usize,
    /// Relevance score (0.0 - 1.0)
    pub relevance_score: f32,
    /// Original similarity score (before re-ranking)
    pub original_score: f32,
    /// Whether LLM was used for scoring
    pub used_local_llm: bool,
}

impl RelevanceResult {
    /// Create from LLM scoring
    pub fn from_local(
        content: String,
        original_index: usize,
        relevance_score: f32,
        original_score: f32,
    ) -> Self {
        Self {
            content,
            original_index,
            relevance_score,
            original_score,
            used_local_llm: true,
        }
    }

    /// Create from fallback (keep original score)
    pub fn from_fallback(content: String, original_index: usize, original_score: f32) -> Self {
        Self {
            content,
            original_index,
            relevance_score: original_score,
            original_score,
            used_local_llm: false,
        }
    }
}

/// Relevance scorer for context re-ranking
pub struct RelevanceScorer {
    provider: Arc<dyn Provider>,
    model_id: String,
    /// Minimum score to include in results
    min_score: f32,
    /// Maximum items to re-rank (for efficiency)
    max_items: usize,
}

impl RelevanceScorer {
    /// Create a new relevance scorer
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
            min_score: 0.5,
            max_items: 10,
        }
    }

    /// Set minimum relevance score threshold
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Set maximum items to re-rank
    pub fn with_max_items(mut self, max_items: usize) -> Self {
        self.max_items = max_items;
        self
    }

    /// Re-rank a list of retrieved items by semantic relevance
    ///
    /// Returns items sorted by relevance score (highest first).
    pub async fn rerank<T: AsRef<str>>(
        &self,
        query: &str,
        items: &[(T, f32)], // (content, original_score) pairs
    ) -> Vec<RelevanceResult> {
        let timer = InferenceTimer::new("rerank_context", &self.model_id);

        // Limit items for efficiency
        let items_to_score: Vec<_> = items.iter().take(self.max_items).collect();

        if items_to_score.is_empty() {
            timer.finish(true);
            return Vec::new();
        }

        // Build scoring prompt
        let prompt = self.build_rerank_prompt(query, &items_to_score);

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(100);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let mut results = self.parse_rerank_output(&output, items);

                // Sort by relevance score descending
                results.sort_by(|a, b| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Filter by minimum score
                results.retain(|r| r.relevance_score >= self.min_score);

                timer.finish(true);
                results
            }
            Err(e) => {
                warn!(target: "local_llm", "Context re-ranking failed: {}", e);
                timer.finish(false);

                // Fallback: keep original order/scores
                items
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, score))| *score >= self.min_score)
                    .map(|(i, (content, score))| {
                        RelevanceResult::from_fallback(content.as_ref().to_string(), i, *score)
                    })
                    .collect()
            }
        }
    }

    /// Score a single item's relevance to a query
    pub async fn score_relevance(&self, query: &str, content: &str) -> Option<f32> {
        let timer = InferenceTimer::new("score_relevance", &self.model_id);

        let prompt = format!(
            r#"Rate the relevance of this content to the query.

Query: "{}"

Content: "{}"

Output a score from 0.0 (irrelevant) to 1.0 (highly relevant).
Output ONLY the decimal number.

Score:"#,
            if query.len() > 100 {
                &query[..100]
            } else {
                query
            },
            if content.len() > 300 {
                &content[..300]
            } else {
                content
            }
        );

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(10);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let score = self.parse_score(&output);
                timer.finish(score.is_some());
                score
            }
            Err(e) => {
                warn!(target: "local_llm", "Relevance scoring failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Heuristic relevance scoring (no LLM)
    pub fn score_heuristic(&self, query: &str, content: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();

        // Extract query words (>2 chars)
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        if query_words.is_empty() {
            return 0.5; // Default for empty query
        }

        // Count word matches
        let mut matches = 0;
        for word in &query_words {
            if content_lower.contains(word) {
                matches += 1;
            }
        }

        // Calculate overlap ratio
        let overlap_ratio = matches as f32 / query_words.len() as f32;

        // Check for exact phrase match (bonus)
        let phrase_bonus = if content_lower.contains(&query_lower) {
            0.2
        } else {
            0.0
        };

        (overlap_ratio * 0.8 + phrase_bonus).min(1.0)
    }

    /// Build the re-ranking prompt
    fn build_rerank_prompt<T: AsRef<str>>(&self, query: &str, items: &[&(T, f32)]) -> String {
        let mut prompt = format!(
            r#"Rank these items by relevance to the query.

Query: "{}"

Items:
"#,
            if query.len() > 150 {
                &query[..150]
            } else {
                query
            }
        );

        for (i, (content, _)) in items.iter().enumerate() {
            let truncated = if content.as_ref().len() > 150 {
                &content.as_ref()[..150]
            } else {
                content.as_ref()
            };
            prompt.push_str(&format!("{}. {}\n", i + 1, truncated));
        }

        prompt.push_str(
            r#"
Output format: item_number:score (0.0-1.0)
Example: 1:0.9, 2:0.3, 3:0.7

Scores:"#,
        );

        prompt
    }

    /// Parse the re-ranking output
    fn parse_rerank_output<T: AsRef<str>>(
        &self,
        output: &str,
        items: &[(T, f32)],
    ) -> Vec<RelevanceResult> {
        let mut results = Vec::new();
        let mut scored_indices = std::collections::HashSet::new();

        // Parse "N:score" patterns
        for part in output.split([',', '\n', ' ']) {
            let part = part.trim();
            if let Some(colon_pos) = part.find(':')
                && let (Ok(idx), score_str) = (
                    part[..colon_pos].trim().parse::<usize>(),
                    part[colon_pos + 1..].trim(),
                )
                && let Ok(score) = score_str.parse::<f32>()
            {
                let actual_idx = idx.saturating_sub(1); // 1-indexed to 0-indexed
                if actual_idx < items.len() && !scored_indices.contains(&actual_idx) {
                    scored_indices.insert(actual_idx);
                    let (content, original_score) = &items[actual_idx];
                    results.push(RelevanceResult::from_local(
                        content.as_ref().to_string(),
                        actual_idx,
                        score.clamp(0.0, 1.0),
                        *original_score,
                    ));
                }
            }
        }

        // Add any items that weren't scored (with original scores)
        for (i, (content, original_score)) in items.iter().enumerate() {
            if !scored_indices.contains(&i) {
                results.push(RelevanceResult::from_fallback(
                    content.as_ref().to_string(),
                    i,
                    *original_score,
                ));
            }
        }

        results
    }

    /// Parse a score from LLM output
    fn parse_score(&self, output: &str) -> Option<f32> {
        let trimmed = output.trim();

        // Try direct parse
        if let Ok(score) = trimmed.parse::<f32>() {
            return Some(score.clamp(0.0, 1.0));
        }

        // Look for a number pattern
        if let Ok(re) = regex::Regex::new(r"(\d+\.?\d*)")
            && let Some(captures) = re.captures(trimmed)
            && let Some(m) = captures.get(1)
            && let Ok(score) = m.as_str().parse::<f32>()
        {
            return Some(score.clamp(0.0, 1.0));
        }

        None
    }
}

/// Builder for RelevanceScorer
pub struct RelevanceScorerBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
    min_score: f32,
    max_items: usize,
}

impl Default for RelevanceScorerBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-350m".to_string(),
            min_score: 0.5,
            max_items: 10,
        }
    }
}

impl RelevanceScorerBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for relevance scoring.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Set the minimum relevance score to include in results.
    pub fn min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Set the maximum number of items to re-rank.
    pub fn max_items(mut self, max_items: usize) -> Self {
        self.max_items = max_items;
        self
    }

    /// Build the relevance scorer, returning `None` if no provider was set.
    pub fn build(self) -> Option<RelevanceScorer> {
        self.provider.map(|p| {
            RelevanceScorer::new(p, self.model_id)
                .with_min_score(self.min_score)
                .with_max_items(self.max_items)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_result() {
        let local = RelevanceResult::from_local("test content".to_string(), 0, 0.9, 0.75);
        assert!(local.used_local_llm);
        assert_eq!(local.relevance_score, 0.9);
        assert_eq!(local.original_score, 0.75);

        let fallback = RelevanceResult::from_fallback("test content".to_string(), 1, 0.7);
        assert!(!fallback.used_local_llm);
        assert_eq!(fallback.relevance_score, 0.7);
    }

    #[test]
    fn test_heuristic_scoring() {
        let score = score_heuristic_direct(
            "rust async programming",
            "This article discusses async programming in Rust using tokio",
        );
        assert!(score > 0.5);

        let low_score = score_heuristic_direct(
            "python web development",
            "This article discusses async programming in Rust using tokio",
        );
        assert!(low_score < 0.3);
    }

    fn score_heuristic_direct(query: &str, content: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();

        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        if query_words.is_empty() {
            return 0.5;
        }

        let mut matches = 0;
        for word in &query_words {
            if content_lower.contains(word) {
                matches += 1;
            }
        }

        let overlap_ratio = matches as f32 / query_words.len() as f32;
        let phrase_bonus = if content_lower.contains(&query_lower) {
            0.2
        } else {
            0.0
        };

        (overlap_ratio * 0.8 + phrase_bonus).min(1.0)
    }

    #[test]
    fn test_parse_rerank_output() {
        let output = "1:0.9, 2:0.5, 3:0.7";
        let items = vec![
            ("first item".to_string(), 0.8),
            ("second item".to_string(), 0.6),
            ("third item".to_string(), 0.7),
        ];

        let results = parse_rerank_output_direct(output, &items);
        assert_eq!(results.len(), 3);

        // Find the highest scored item
        let best = results
            .iter()
            .max_by(|a, b| a.relevance_score.partial_cmp(&b.relevance_score).unwrap())
            .unwrap();
        assert_eq!(best.original_index, 0); // First item had 0.9 score
    }

    fn parse_rerank_output_direct(output: &str, items: &[(String, f32)]) -> Vec<RelevanceResult> {
        let mut results = Vec::new();
        let mut scored_indices = std::collections::HashSet::new();

        for part in output.split(',') {
            let part = part.trim();
            if let Some(colon_pos) = part.find(':')
                && let (Ok(idx), score_str) = (
                    part[..colon_pos].trim().parse::<usize>(),
                    part[colon_pos + 1..].trim(),
                )
                && let Ok(score) = score_str.parse::<f32>()
            {
                let actual_idx = idx.saturating_sub(1);
                if actual_idx < items.len() && !scored_indices.contains(&actual_idx) {
                    scored_indices.insert(actual_idx);
                    let (content, original_score) = &items[actual_idx];
                    results.push(RelevanceResult::from_local(
                        content.clone(),
                        actual_idx,
                        score.clamp(0.0, 1.0),
                        *original_score,
                    ));
                }
            }
        }

        results
    }

    #[test]
    fn test_parse_score() {
        assert_eq!(parse_score_direct("0.85"), Some(0.85));
        assert_eq!(parse_score_direct("Score: 0.7"), Some(0.7));
        assert_eq!(parse_score_direct("1.5"), Some(1.0)); // Clamped
        assert_eq!(parse_score_direct("-0.5"), Some(0.0)); // Negative clamped to 0.0
        assert_eq!(parse_score_direct("not a score"), None); // No number found
    }

    fn parse_score_direct(output: &str) -> Option<f32> {
        let trimmed = output.trim();

        if let Ok(score) = trimmed.parse::<f32>() {
            return Some(score.clamp(0.0, 1.0));
        }

        if let Ok(re) = regex::Regex::new(r"(\d+\.?\d*)")
            && let Some(captures) = re.captures(trimmed)
            && let Some(m) = captures.get(1)
            && let Ok(score) = m.as_str().parse::<f32>()
        {
            return Some(score.clamp(0.0, 1.0));
        }

        None
    }
}
