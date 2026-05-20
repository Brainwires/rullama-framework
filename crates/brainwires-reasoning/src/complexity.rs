//! Complexity Scorer - Task Complexity Assessment
//!
//! Uses a provider to score task complexity (0.0 - 1.0),
//! enabling adaptive k adjustment in MDAP voting.

use std::sync::Arc;
use tracing::warn;

use brainwires_core::message::Message;
use brainwires_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

/// Result of complexity scoring
#[derive(Clone, Debug)]
pub struct ComplexityResult {
    /// Complexity score (0.0 = trivial, 1.0 = very complex)
    pub score: f32,
    /// Confidence in the score (0.0 - 1.0)
    pub confidence: f32,
    /// Whether LLM was used (vs default)
    pub used_local_llm: bool,
}

impl ComplexityResult {
    /// Create a default complexity result (fallback)
    pub fn default_complexity() -> Self {
        Self {
            score: 0.5, // Medium complexity as default
            confidence: 0.3,
            used_local_llm: false,
        }
    }

    /// Create a result from LLM scoring
    pub fn from_local(score: f32, confidence: f32) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            used_local_llm: true,
        }
    }
}

/// Complexity scorer for task difficulty assessment
pub struct ComplexityScorer {
    provider: Arc<dyn Provider>,
    model_id: String,
}

impl ComplexityScorer {
    /// Create a new complexity scorer
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
        }
    }

    /// Score the complexity of a task description
    ///
    /// Returns a score from 0.0 (trivial) to 1.0 (very complex).
    /// Returns None if scoring fails, allowing fallback to default.
    pub async fn score(&self, task_description: &str) -> Option<ComplexityResult> {
        let timer = InferenceTimer::new("complexity_score", &self.model_id);

        let system_prompt = self.build_scoring_prompt();
        let user_prompt = format!(
            "Rate the complexity of this task from 0.0 (trivial) to 1.0 (very complex). Output ONLY a decimal number.\n\nTask: {}",
            task_description
        );

        let messages = vec![Message::user(&user_prompt)];
        let options = ChatOptions::deterministic(10).system(system_prompt);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let text = response.message.text_or_summary();
                if let Some(score) = self.parse_score(&text) {
                    timer.finish(true);
                    Some(ComplexityResult::from_local(score, 0.8))
                } else {
                    timer.finish(false);
                    None
                }
            }
            Err(e) => {
                warn!(target: "local_llm", "Complexity scoring failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Score complexity synchronously (for use in sync contexts)
    /// Uses heuristics instead of LLM for speed.
    pub fn score_heuristic(&self, task_description: &str) -> ComplexityResult {
        let desc_lower = task_description.to_lowercase();
        let mut score: f32 = 0.3; // Base score

        // Complexity indicators (increase score)
        let complex_indicators = [
            ("multiple", 0.1),
            ("several", 0.1),
            ("complex", 0.15),
            ("difficult", 0.15),
            ("careful", 0.1),
            ("ensure", 0.05),
            ("validate", 0.1),
            ("analyze", 0.1),
            ("refactor", 0.15),
            ("architecture", 0.2),
            ("design", 0.1),
            ("optimize", 0.15),
            ("performance", 0.1),
            ("security", 0.15),
            ("concurrent", 0.2),
            ("async", 0.1),
            ("parallel", 0.15),
            ("distributed", 0.2),
        ];

        // Simplicity indicators (decrease score)
        let simple_indicators = [
            ("simple", -0.1),
            ("trivial", -0.15),
            ("just", -0.05),
            ("only", -0.05),
            ("basic", -0.1),
            ("single", -0.05),
            ("one", -0.05),
            ("quick", -0.1),
            ("easy", -0.1),
        ];

        for (keyword, adjustment) in complex_indicators {
            if desc_lower.contains(keyword) {
                score += adjustment;
            }
        }

        for (keyword, adjustment) in simple_indicators {
            if desc_lower.contains(keyword) {
                score += adjustment;
            }
        }

        // Length-based adjustment (longer = more complex)
        let word_count = task_description.split_whitespace().count();
        if word_count > 50 {
            score += 0.15;
        } else if word_count > 30 {
            score += 0.1;
        } else if word_count < 10 {
            score -= 0.1;
        }

        ComplexityResult {
            score: score.clamp(0.0, 1.0),
            confidence: 0.4, // Lower confidence for heuristic
            used_local_llm: false,
        }
    }

    /// Build the system prompt for complexity scoring
    fn build_scoring_prompt(&self) -> String {
        r#"You are a task complexity evaluator. Given a task description, output a complexity score.

Scoring guide:
- 0.0-0.2: Trivial (single step, no decisions)
- 0.2-0.4: Simple (few steps, straightforward)
- 0.4-0.6: Moderate (multiple steps, some decisions)
- 0.6-0.8: Complex (many steps, careful reasoning needed)
- 0.8-1.0: Very complex (intricate logic, multiple dependencies)

Consider:
- Number of steps or operations needed
- Required reasoning depth
- Ambiguity in requirements
- Dependencies between parts
- Potential for errors

Output ONLY a decimal number between 0.0 and 1.0."#
            .to_string()
    }

    /// Parse the LLM output to extract a score
    fn parse_score(&self, output: &str) -> Option<f32> {
        // Try to find a floating point number in the output
        let cleaned = output.trim();

        // Direct parse
        if let Ok(score) = cleaned.parse::<f32>() {
            return Some(score.clamp(0.0, 1.0));
        }

        // Look for a number pattern
        let number_pattern = regex::Regex::new(r"(\d+\.?\d*)").ok()?;
        if let Some(captures) = number_pattern.captures(cleaned)
            && let Some(m) = captures.get(1)
            && let Ok(score) = m.as_str().parse::<f32>()
        {
            return Some(score.clamp(0.0, 1.0));
        }

        None
    }
}

/// Builder for ComplexityScorer
pub struct ComplexityScorerBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
}

impl Default for ComplexityScorerBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-350m".to_string(),
        }
    }
}

impl ComplexityScorerBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for complexity scoring.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Build the complexity scorer, returning `None` if no provider was set.
    pub fn build(self) -> Option<ComplexityScorer> {
        self.provider
            .map(|p| ComplexityScorer::new(p, self.model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_result_default() {
        let result = ComplexityResult::default_complexity();
        assert_eq!(result.score, 0.5);
        assert!(!result.used_local_llm);
    }

    #[test]
    fn test_complexity_result_clamping() {
        let result = ComplexityResult::from_local(1.5, 0.9);
        assert_eq!(result.score, 1.0); // Clamped

        let result = ComplexityResult::from_local(-0.5, 0.9);
        assert_eq!(result.score, 0.0); // Clamped
    }

    #[test]
    fn test_heuristic_scoring() {
        // Create a stub scorer for testing heuristics
        let _scorer = ComplexityScorerBuilder::default();

        // Test with a simple task
        let simple = "read a file";
        let simple_score = score_heuristic_direct(simple);
        assert!(simple_score < 0.5);

        // Test with a complex task
        let complex = "refactor the architecture to implement a distributed concurrent system with multiple parallel workers";
        let complex_score = score_heuristic_direct(complex);
        assert!(complex_score > 0.5);
    }

    // Helper for testing heuristic scoring
    fn score_heuristic_direct(task: &str) -> f32 {
        let desc_lower = task.to_lowercase();
        let mut score: f32 = 0.3;

        let complex_indicators = [
            ("multiple", 0.1),
            ("complex", 0.15),
            ("refactor", 0.15),
            ("architecture", 0.2),
            ("concurrent", 0.2),
            ("parallel", 0.15),
            ("distributed", 0.2),
        ];

        let simple_indicators = [("simple", -0.1), ("just", -0.05), ("basic", -0.1)];

        for (keyword, adjustment) in complex_indicators {
            if desc_lower.contains(keyword) {
                score += adjustment;
            }
        }

        for (keyword, adjustment) in simple_indicators {
            if desc_lower.contains(keyword) {
                score += adjustment;
            }
        }

        score.clamp(0.0, 1.0)
    }

    #[test]
    fn test_parse_score() {
        let _scorer = ComplexityScorerBuilder::default();

        // Test parsing logic
        assert_eq!(parse_score_direct("0.5"), Some(0.5));
        assert_eq!(parse_score_direct("0.85"), Some(0.85));
        assert_eq!(parse_score_direct("The complexity is 0.7"), Some(0.7));
        assert_eq!(parse_score_direct("1.5"), Some(1.0)); // Clamped
    }

    fn parse_score_direct(output: &str) -> Option<f32> {
        let cleaned = output.trim();
        if let Ok(score) = cleaned.parse::<f32>() {
            return Some(score.clamp(0.0, 1.0));
        }
        let number_pattern = regex::Regex::new(r"(\d+\.?\d*)").ok()?;
        if let Some(captures) = number_pattern.captures(cleaned)
            && let Some(m) = captures.get(1)
            && let Ok(score) = m.as_str().parse::<f32>()
        {
            return Some(score.clamp(0.0, 1.0));
        }
        None
    }
}
