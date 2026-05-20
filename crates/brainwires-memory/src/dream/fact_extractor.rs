//! Fact extraction — use an LLM to distil durable facts from conversation
//! summaries for long-term cold-tier storage.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use brainwires_core::{ChatOptions, Message, Provider};

/// A single durable fact extracted from a conversation summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// The fact text.
    pub content: String,
    /// Semantic category of the fact.
    pub category: FactCategory,
    /// Confidence that this fact is accurate (0.0–1.0).
    pub confidence: f32,
}

/// Semantic category for an extracted fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    /// A stated user preference (e.g. "prefers dark mode").
    UserPreference,
    /// A detail about the user's project or domain.
    ProjectDetail,
    /// A recurring behavioural pattern.
    BehavioralPattern,
    /// Knowledge about a tool or its usage.
    ToolKnowledge,
    /// A technical decision that was made.
    TechnicalDecision,
}

/// Stateless helper that calls an LLM to extract facts from a summary.
pub struct FactExtractor;

impl FactExtractor {
    /// Extract durable facts from the given summary text.
    ///
    /// Uses `provider` to call the LLM with a structured prompt asking it to
    /// return facts as a JSON array.
    pub async fn extract_facts(
        summary: &str,
        provider: &dyn Provider,
    ) -> Result<Vec<ExtractedFact>> {
        let prompt = format!(
            "You are a knowledge extractor. Given the following conversation summary, \
             extract durable facts that would be useful to remember long-term.\n\n\
             For each fact, provide:\n\
             - \"content\": the fact text\n\
             - \"category\": one of: user_preference, project_detail, behavioral_pattern, \
               tool_knowledge, technical_decision\n\
             - \"confidence\": a float 0.0-1.0\n\n\
             Return ONLY a JSON array of objects. No markdown fences.\n\n\
             Summary:\n{summary}"
        );

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions {
            temperature: Some(0.2),
            max_tokens: Some(2048),
            ..Default::default()
        };

        let response = provider.chat(&messages, None, &options).await?;
        let text = response.message.text_or_summary();

        // Try to parse the JSON array from the response.
        let facts: Vec<ExtractedFact> = serde_json::from_str(text.trim()).unwrap_or_else(|_| {
            // Fallback: if parsing fails, create a single fact from the raw text.
            tracing::warn!("Failed to parse fact extraction JSON; creating fallback fact");
            vec![ExtractedFact {
                content: text.trim().to_string(),
                category: FactCategory::ProjectDetail,
                confidence: 0.5,
            }]
        });

        Ok(facts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_category_serde_roundtrip() {
        let categories = vec![
            FactCategory::UserPreference,
            FactCategory::ProjectDetail,
            FactCategory::BehavioralPattern,
            FactCategory::ToolKnowledge,
            FactCategory::TechnicalDecision,
        ];
        for cat in categories {
            let json = serde_json::to_string(&cat).unwrap();
            let parsed: FactCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, cat);
        }
    }

    #[test]
    fn test_extracted_fact_serde_roundtrip() {
        let fact = ExtractedFact {
            content: "User prefers Rust over Python".to_string(),
            category: FactCategory::UserPreference,
            confidence: 0.9,
        };
        let json = serde_json::to_string(&fact).unwrap();
        let parsed: ExtractedFact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, fact.content);
        assert_eq!(parsed.category, FactCategory::UserPreference);
        assert!((parsed.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fact_category_json_names() {
        assert_eq!(
            serde_json::to_string(&FactCategory::UserPreference).unwrap(),
            "\"user_preference\""
        );
        assert_eq!(
            serde_json::to_string(&FactCategory::TechnicalDecision).unwrap(),
            "\"technical_decision\""
        );
    }
}
