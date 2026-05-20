//! Response Confidence Extraction
//!
//! Based on CISC paper (arxiv:2502.06233v1) - extracts confidence scores from
//! LLM responses based on multiple heuristics for use in decision-making and SEAL learning.

use crate::ChatResponse;

/// Response confidence metrics
#[derive(Debug, Clone, Default)]
pub struct ResponseConfidence {
    /// Overall confidence score (0.0 - 1.0)
    pub score: f64,
    /// Individual factors that contributed to the score
    pub factors: ConfidenceFactors,
}

impl ResponseConfidence {
    /// Check if this is considered a high-confidence response
    pub fn is_high_confidence(&self) -> bool {
        self.score >= 0.8
    }

    /// Check if this is considered a low-confidence response
    pub fn is_low_confidence(&self) -> bool {
        self.score < 0.6
    }

    /// Get a human-readable confidence level
    pub fn level(&self) -> &'static str {
        if self.score >= 0.9 {
            "very_high"
        } else if self.score >= 0.8 {
            "high"
        } else if self.score >= 0.6 {
            "medium"
        } else if self.score >= 0.4 {
            "low"
        } else {
            "very_low"
        }
    }
}

/// Individual factors that contribute to confidence score
#[derive(Debug, Clone, Default)]
pub struct ConfidenceFactors {
    /// Based on finish_reason (stop = high, truncated = low)
    pub completion_confidence: f64,
    /// Based on hedging/uncertainty patterns in text
    pub pattern_confidence: f64,
    /// Based on response length (normalized)
    pub length_confidence: f64,
    /// Based on presence of tool use (structured = higher confidence)
    pub structure_confidence: f64,
}

impl ConfidenceFactors {
    /// Get the factor with the lowest confidence
    pub fn weakest_factor(&self) -> (&'static str, f64) {
        let factors = [
            ("completion", self.completion_confidence),
            ("pattern", self.pattern_confidence),
            ("length", self.length_confidence),
            ("structure", self.structure_confidence),
        ];

        factors
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("unknown", 0.5))
    }
}

/// Patterns that indicate low confidence (hedging language)
const LOW_CONFIDENCE_PATTERNS: &[&str] = &[
    "i'm not sure",
    "i think",
    "possibly",
    "might be",
    "could be",
    "i believe",
    "probably",
    "perhaps",
    "maybe",
    "not certain",
    "unclear",
    "i guess",
    "it seems",
    "apparently",
];

/// Patterns that indicate self-correction (can reduce confidence)
const SELF_CORRECTION_PATTERNS: &[&str] = &[
    "wait,",
    "actually,",
    "let me reconsider",
    "i made a mistake",
    "correction:",
    "i was wrong",
    "on second thought",
    "i need to revise",
    "let me correct",
    "that's not right",
];

/// Patterns that indicate high confidence assertions
const HIGH_CONFIDENCE_PATTERNS: &[&str] = &[
    "the answer is",
    "definitely",
    "certainly",
    "clearly",
    "without doubt",
    "the solution is",
    "this will work",
    "i can confirm",
];

/// Extract confidence from a chat response
///
/// Analyzes the response using multiple heuristics:
/// 1. Completion status (finish_reason)
/// 2. Language patterns (hedging, self-correction, assertions)
/// 3. Response length (too short or too long can indicate issues)
/// 4. Structure (tool use vs pure text)
pub fn extract_confidence(response: &ChatResponse) -> ResponseConfidence {
    // Get text content for analysis
    let text = get_response_text(response);

    // 1. Completion confidence (based on finish_reason)
    let completion_confidence = calculate_completion_confidence(&response.finish_reason);

    // 2. Pattern confidence (based on hedging/assertion language)
    let pattern_confidence = calculate_pattern_confidence(&text);

    // 3. Length confidence (optimal range analysis)
    let length_confidence = calculate_length_confidence(&text);

    // 4. Structure confidence (tool use indicates structured thinking)
    let structure_confidence = calculate_structure_confidence(response);

    // Weighted average with emphasis on pattern and completion
    let score = completion_confidence * 0.30
        + pattern_confidence * 0.35
        + length_confidence * 0.15
        + structure_confidence * 0.20;

    ResponseConfidence {
        score: score.clamp(0.0, 1.0),
        factors: ConfidenceFactors {
            completion_confidence,
            pattern_confidence,
            length_confidence,
            structure_confidence,
        },
    }
}

/// Extract text from response (handles both simple text and blocks)
fn get_response_text(response: &ChatResponse) -> String {
    use crate::MessageContent;

    match &response.message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Blocks(blocks) => {
            use crate::ContentBlock;
            blocks
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

/// Calculate confidence based on completion reason
fn calculate_completion_confidence(finish_reason: &Option<String>) -> f64 {
    match finish_reason.as_deref() {
        Some("stop") | Some("end_turn") => 0.95,
        Some("tool_use") => 0.90, // Structured response with tool usage
        Some("length") | Some("max_tokens") => 0.50, // Truncated = lower confidence
        Some("content_filter") => 0.30, // Content was filtered
        None => 0.70,             // Unknown status
        _ => 0.60,                // Other reasons
    }
}

/// Calculate confidence based on language patterns in text
fn calculate_pattern_confidence(text: &str) -> f64 {
    let text_lower = text.to_lowercase();

    // Count low confidence patterns
    let low_confidence_count = LOW_CONFIDENCE_PATTERNS
        .iter()
        .filter(|p| text_lower.contains(*p))
        .count();

    // Count self-correction patterns (weight more heavily)
    let self_correction_count = SELF_CORRECTION_PATTERNS
        .iter()
        .filter(|p| text_lower.contains(*p))
        .count();

    // Count high confidence patterns
    let high_confidence_count = HIGH_CONFIDENCE_PATTERNS
        .iter()
        .filter(|p| text_lower.contains(*p))
        .count();

    // Start with baseline confidence
    let mut confidence = 0.75;

    // Reduce for hedging language (diminishing returns)
    confidence -= (low_confidence_count as f64 * 0.08).min(0.35);

    // Reduce more for self-correction (indicates uncertainty)
    confidence -= (self_correction_count as f64 * 0.15).min(0.30);

    // Boost for confident assertions (smaller boost)
    confidence += (high_confidence_count as f64 * 0.05).min(0.15);

    confidence.clamp(0.25, 0.98)
}

/// Calculate confidence based on response length
///
/// Optimal length is between 50-500 tokens (estimated by chars/4).
/// Too short may indicate incomplete thinking.
/// Too long may indicate rambling or uncertainty.
fn calculate_length_confidence(text: &str) -> f64 {
    // Estimate token count (rough approximation)
    let token_estimate = text.len() / 4;

    if token_estimate < 10 {
        0.40 // Very short - possibly incomplete
    } else if token_estimate < 30 {
        0.60 // Short but might be appropriate for simple queries
    } else if token_estimate < 50 {
        0.75 // Below optimal but reasonable
    } else if token_estimate <= 500 {
        0.90 // Optimal range
    } else if token_estimate <= 1000 {
        0.75 // Getting long but still reasonable
    } else if token_estimate <= 2000 {
        0.60 // Very long, might be over-explaining
    } else {
        0.50 // Extremely long, likely rambling
    }
}

/// Calculate confidence based on response structure
///
/// Tool use indicates structured thinking and specific actions,
/// which often correlates with higher confidence and accuracy.
fn calculate_structure_confidence(response: &ChatResponse) -> f64 {
    use crate::MessageContent;

    match &response.message.content {
        MessageContent::Text(_) => 0.70, // Pure text
        MessageContent::Blocks(blocks) => {
            use crate::ContentBlock;

            // Check for tool use blocks
            let has_tool_use = blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

            if has_tool_use {
                0.90 // Structured response with tools
            } else {
                0.75 // Multiple blocks but no tools
            }
        }
    }
}

/// Quick confidence check without full analysis
///
/// Useful for early detection of low-confidence responses.
pub fn quick_confidence_check(response: &ChatResponse) -> bool {
    // Check finish reason
    if response.finish_reason.as_deref() == Some("length") {
        return false;
    }

    // Check for obvious low-confidence patterns
    let text = get_response_text(response);
    let text_lower = text.to_lowercase();

    // Quick pattern scan
    let obvious_low_confidence = [
        "i'm not sure",
        "i don't know",
        "i cannot",
        "i made a mistake",
        "that's not right",
    ];

    !obvious_low_confidence
        .iter()
        .any(|p| text_lower.contains(*p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, MessageContent, Usage};

    fn make_response(text: &str, finish_reason: Option<&str>) -> ChatResponse {
        ChatResponse {
            message: Message {
                role: crate::Role::Assistant,
                content: MessageContent::Text(text.to_string()),
                name: None,
                metadata: None,
            },
            usage: Usage::default(),
            finish_reason: finish_reason.map(String::from),
        }
    }

    #[test]
    fn test_high_confidence_response() {
        let response = make_response(
            "The solution is to use a hashmap for O(1) lookup. This will definitely work.",
            Some("stop"),
        );
        let confidence = extract_confidence(&response);

        assert!(confidence.score > 0.75);
        assert!(confidence.is_high_confidence() || confidence.score >= 0.7);
    }

    #[test]
    fn test_low_confidence_response() {
        let response = make_response(
            "I'm not sure, but I think maybe this could possibly work. Let me reconsider...",
            Some("stop"),
        );
        let confidence = extract_confidence(&response);

        // This text has 5+ hedging patterns + self-correction, should be lower than high-confidence
        assert!(
            confidence.score < 0.75,
            "Expected low confidence score, got {}",
            confidence.score
        );
        // At minimum, should have lower pattern confidence
        assert!(confidence.factors.pattern_confidence < 0.7);
    }

    #[test]
    fn test_truncated_response() {
        let response = make_response(
            "The answer involves several steps. First, we need to",
            Some("length"),
        );
        let confidence = extract_confidence(&response);

        assert!(confidence.factors.completion_confidence < 0.6);
    }

    #[test]
    fn test_very_short_response() {
        let response = make_response("Yes", Some("stop"));
        let confidence = extract_confidence(&response);

        assert!(confidence.factors.length_confidence < 0.7);
    }

    #[test]
    fn test_pattern_confidence_calculation() {
        // High confidence text
        let high = calculate_pattern_confidence(
            "The solution is definitely correct and will certainly work.",
        );
        assert!(high > 0.7);

        // Low confidence text
        let low =
            calculate_pattern_confidence("I'm not sure, but maybe it could possibly work perhaps.");
        assert!(low < 0.6);
    }

    #[test]
    fn test_quick_confidence_check() {
        let good = make_response("Here is the implementation you need.", Some("stop"));
        assert!(quick_confidence_check(&good));

        let bad = make_response("I don't know how to do this.", Some("stop"));
        assert!(!quick_confidence_check(&bad));
    }

    #[test]
    fn test_confidence_level() {
        let high = ResponseConfidence {
            score: 0.9,
            ..Default::default()
        };
        assert_eq!(high.level(), "very_high");

        let low = ResponseConfidence {
            score: 0.3,
            ..Default::default()
        };
        assert_eq!(low.level(), "very_low");
    }
}
