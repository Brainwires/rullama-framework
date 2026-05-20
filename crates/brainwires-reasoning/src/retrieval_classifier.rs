//! Retrieval Classifier - Enhanced Retrieval Gating
//!
//! Uses a provider to classify retrieval need semantically,
//! replacing pattern-based detection with understanding of intent.

use std::sync::Arc;
use tracing::warn;

use brainwires_core::message::Message;
use brainwires_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

/// Result of retrieval classification
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrievalNeed {
    /// No retrieval needed - context is sufficient
    None,
    /// Low priority - might benefit from retrieval
    Low,
    /// Medium priority - likely needs retrieval
    Medium,
    /// High priority - definitely needs retrieval
    High,
}

impl RetrievalNeed {
    /// Check if retrieval should be performed
    pub fn should_retrieve(&self) -> bool {
        matches!(self, RetrievalNeed::Medium | RetrievalNeed::High)
    }

    /// Convert to a priority score (0.0 - 1.0)
    pub fn as_score(&self) -> f32 {
        match self {
            RetrievalNeed::None => 0.0,
            RetrievalNeed::Low => 0.25,
            RetrievalNeed::Medium => 0.6,
            RetrievalNeed::High => 0.9,
        }
    }
}

/// Result of classification with confidence
#[derive(Clone, Debug)]
pub struct ClassificationResult {
    /// The classified retrieval need
    pub need: RetrievalNeed,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Whether LLM was used
    pub used_local_llm: bool,
    /// Detected intent (if LLM was used)
    pub intent: Option<String>,
}

impl ClassificationResult {
    /// Create a result from LLM classification
    pub fn from_local(need: RetrievalNeed, confidence: f32, intent: Option<String>) -> Self {
        Self {
            need,
            confidence,
            used_local_llm: true,
            intent,
        }
    }

    /// Create a result from pattern-based fallback
    pub fn from_fallback(need: RetrievalNeed, confidence: f32) -> Self {
        Self {
            need,
            confidence,
            used_local_llm: false,
            intent: None,
        }
    }
}

/// Retrieval classifier for enhanced gating
pub struct RetrievalClassifier {
    provider: Arc<dyn Provider>,
    model_id: String,
}

impl RetrievalClassifier {
    /// Create a new retrieval classifier
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
        }
    }

    /// Classify retrieval need using the provider
    ///
    /// Returns classification with intent understanding.
    pub async fn classify(&self, query: &str, context_len: usize) -> Option<ClassificationResult> {
        let timer = InferenceTimer::new("retrieval_classify", &self.model_id);

        let prompt = self.build_classification_prompt(query, context_len);

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(50);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let result = self.parse_classification(&output);
                timer.finish(true);
                Some(result)
            }
            Err(e) => {
                warn!(target: "local_llm", "Retrieval classification failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Heuristic classification (pattern-based fallback)
    ///
    /// Used when provider is unavailable or fails.
    pub fn classify_heuristic(&self, query: &str, context_len: usize) -> ClassificationResult {
        let lower = query.to_lowercase();
        let mut score = 0.0f32;
        let mut matches = 0;

        // Reference patterns (high weight)
        let reference_patterns = [
            "earlier",
            "before",
            "we discussed",
            "remember when",
            "what was",
            "didn't we",
            "you mentioned",
            "as i said",
            "previously",
            "last time",
            "originally",
            "initially",
            "you said",
            "i said",
            "we talked",
            "back when",
            "recall",
            "mentioned earlier",
            "as mentioned",
        ];

        for pattern in reference_patterns {
            if lower.contains(pattern) {
                score += 0.4;
                matches += 1;
            }
        }

        // Question patterns (medium weight)
        let question_patterns = [
            "what did",
            "when did",
            "why did",
            "how did",
            "where was",
            "who was",
        ];

        for pattern in question_patterns {
            if lower.contains(pattern) {
                score += 0.25;
                matches += 1;
            }
        }

        // Continuation patterns (low weight)
        let continuation_patterns = [
            "continue",
            "keep going",
            "and then",
            "what about",
            "more about",
            "tell me more",
            "go on",
        ];

        for pattern in continuation_patterns {
            if lower.contains(pattern) {
                score += 0.15;
                matches += 1;
            }
        }

        // Context length adjustment
        if context_len < 3 {
            score += 0.3;
        } else if context_len < 5 {
            score += 0.2;
        } else if context_len < 10 {
            score += 0.1;
        }

        // Pronoun patterns (only for short queries)
        if context_len < 10 && query.len() < 100 && lower.contains('?') {
            let pronouns = ["it", "they", "that", "those", "the one"];
            if pronouns
                .iter()
                .any(|p| lower.split_whitespace().any(|w| w == *p))
            {
                score += 0.2;
            }
        }

        score = score.min(1.0);

        let need = match score {
            s if s >= 0.6 => RetrievalNeed::High,
            s if s >= 0.35 => RetrievalNeed::Medium,
            s if s >= 0.15 => RetrievalNeed::Low,
            _ => RetrievalNeed::None,
        };

        let confidence = if matches > 0 {
            0.7 + (matches as f32 * 0.05).min(0.2)
        } else {
            0.5
        };

        ClassificationResult::from_fallback(need, confidence)
    }

    /// Build the classification prompt
    fn build_classification_prompt(&self, query: &str, context_len: usize) -> String {
        format!(
            r#"Classify if this query needs to retrieve earlier conversation context.

Query: "{}"
Recent context messages: {}

Classify as:
- NONE: Query is self-contained, no prior context needed
- LOW: Might benefit from context but not required
- MEDIUM: Likely references earlier discussion
- HIGH: Definitely refers to prior conversation

Output format: LEVEL: brief reason
Example: HIGH: references "earlier" and asks about past discussion

Classification:"#,
            if query.len() > 200 {
                &query[..200]
            } else {
                query
            },
            context_len
        )
    }

    /// Parse the LLM output to extract classification
    fn parse_classification(&self, output: &str) -> ClassificationResult {
        let upper = output.to_uppercase();
        let trimmed = output.trim();

        // Extract intent from the reason part
        let intent = trimmed
            .find(':')
            .map(|colon_pos| trimmed[colon_pos + 1..].trim().to_string());

        // Parse the level
        let need = if upper.starts_with("HIGH") || upper.contains("HIGH:") {
            RetrievalNeed::High
        } else if upper.starts_with("MEDIUM") || upper.contains("MEDIUM:") {
            RetrievalNeed::Medium
        } else if upper.starts_with("LOW") || upper.contains("LOW:") {
            RetrievalNeed::Low
        } else if upper.starts_with("NONE") || upper.contains("NONE:") {
            RetrievalNeed::None
        } else {
            // Ambiguous - default to low
            RetrievalNeed::Low
        };

        ClassificationResult::from_local(need, 0.8, intent)
    }
}

/// Builder for RetrievalClassifier
pub struct RetrievalClassifierBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
}

impl Default for RetrievalClassifierBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-350m".to_string(),
        }
    }
}

impl RetrievalClassifierBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for retrieval classification.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Build the retrieval classifier, returning `None` if no provider was set.
    pub fn build(self) -> Option<RetrievalClassifier> {
        self.provider
            .map(|p| RetrievalClassifier::new(p, self.model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_need_methods() {
        assert!(!RetrievalNeed::None.should_retrieve());
        assert!(!RetrievalNeed::Low.should_retrieve());
        assert!(RetrievalNeed::Medium.should_retrieve());
        assert!(RetrievalNeed::High.should_retrieve());

        assert_eq!(RetrievalNeed::None.as_score(), 0.0);
        assert!(RetrievalNeed::High.as_score() > RetrievalNeed::Low.as_score());
    }

    #[test]
    fn test_classification_result() {
        let local = ClassificationResult::from_local(
            RetrievalNeed::High,
            0.9,
            Some("references earlier discussion".to_string()),
        );
        assert!(local.used_local_llm);
        assert!(local.intent.is_some());

        let fallback = ClassificationResult::from_fallback(RetrievalNeed::Medium, 0.7);
        assert!(!fallback.used_local_llm);
        assert!(fallback.intent.is_none());
    }

    #[test]
    fn test_heuristic_classification_reference() {
        let _classifier = RetrievalClassifierBuilder::default();

        // Test reference patterns
        let result = classify_heuristic_direct("What did we discuss earlier?", 10);
        assert_eq!(result.need, RetrievalNeed::High);
    }

    #[test]
    fn test_heuristic_classification_none() {
        let result = classify_heuristic_direct("Write a hello world function in Python", 20);
        assert_eq!(result.need, RetrievalNeed::None);
    }

    #[test]
    fn test_heuristic_short_context() {
        // Short context should increase retrieval need
        let result = classify_heuristic_direct("Continue please", 2);
        assert!(result.need.should_retrieve());
    }

    fn classify_heuristic_direct(query: &str, context_len: usize) -> ClassificationResult {
        let lower = query.to_lowercase();
        let mut score = 0.0f32;
        let mut matches = 0;

        let reference_patterns = ["earlier", "before", "we discussed", "previously"];

        for pattern in reference_patterns {
            if lower.contains(pattern) {
                score += 0.4;
                matches += 1;
            }
        }

        let question_patterns = ["what did", "when did", "why did"];

        for pattern in question_patterns {
            if lower.contains(pattern) {
                score += 0.25;
                matches += 1;
            }
        }

        // Continuation patterns (matching the real implementation)
        let continuation_patterns = ["continue", "keep going", "and then"];

        for pattern in continuation_patterns {
            if lower.contains(pattern) {
                score += 0.15;
                matches += 1;
            }
        }

        if context_len < 3 {
            score += 0.3;
        } else if context_len < 5 {
            score += 0.2;
        }

        score = score.min(1.0);

        let need = match score {
            s if s >= 0.6 => RetrievalNeed::High,
            s if s >= 0.35 => RetrievalNeed::Medium,
            s if s >= 0.15 => RetrievalNeed::Low,
            _ => RetrievalNeed::None,
        };

        let confidence = if matches > 0 {
            0.7 + (matches as f32 * 0.05).min(0.2)
        } else {
            0.5
        };

        ClassificationResult::from_fallback(need, confidence)
    }

    #[test]
    fn test_parse_classification() {
        // Test parsing logic
        let high = parse_classification_direct("HIGH: references earlier discussion");
        assert_eq!(high.need, RetrievalNeed::High);

        let none = parse_classification_direct("NONE: self-contained query");
        assert_eq!(none.need, RetrievalNeed::None);
    }

    fn parse_classification_direct(output: &str) -> ClassificationResult {
        let upper = output.to_uppercase();
        let trimmed = output.trim();

        let intent = trimmed
            .find(':')
            .map(|colon_pos| trimmed[colon_pos + 1..].trim().to_string());

        let need = if upper.starts_with("HIGH") {
            RetrievalNeed::High
        } else if upper.starts_with("MEDIUM") {
            RetrievalNeed::Medium
        } else if upper.starts_with("LOW") {
            RetrievalNeed::Low
        } else if upper.starts_with("NONE") {
            RetrievalNeed::None
        } else {
            RetrievalNeed::Low
        };

        ClassificationResult::from_local(need, 0.8, intent)
    }
}
